import atexit
import importlib
import pickle
from collections.abc import Callable
from concurrent import futures
from typing import TypeVar, Generic, Type, Protocol, List, Dict

import grpc
from quake.distributedwrapper import rwrap_pb2_grpc
from quake.distributedwrapper.rwrap_pb2 import (
    CommandRequest,
    CommandResponse,
    InstanceResponse,
    InstanceRequest,
    ImportResponse,
    ImportRequest,
    CleanupResponse,
    CleanupRequest,
)

MAX_MESSAGE_LENGTH = 1024 * 1024 * 1024

T = TypeVar("T")


def clean():
    for obj in Local._objects:
        obj._stub.SendCleanup(CleanupRequest())
    for channel in Local._connections.values():
        channel.close()


atexit.register(clean)


class IndirectLocal:
    pass


class LocalVersion(Protocol):
    def import_module(self, package, as_name=None, item=None): ...

    def instantiate(self, *arguments, **keywords): ...


class Local:
    _connections = {}
    _functions = set()
    _objects: List["Local"] = []
    _uuid_lookup: Dict[int, "Local"] = {}
    _internal_attrs = {
        "_special_function",
        "_internal_attrs",
        "_connections",
        "_connection",
        "_objects",
        "_uuid_lookup",
        "_cls",
        "_stub",
        "uuid",
        "_address",
        "instantiate",
        "establish_connection",
        "_interceptor",
        "import_module",
        "_adjust_for_nonlocal",
        "register_function",
        "_functions",
        "_decode_response",
    }

    def __init__(self, address: str, cls: Type[T]):
        self._address = address
        self._special_function = self._interceptor
        self._connection = Local.establish_connection(address)
        self._cls = cls
        self._stub = rwrap_pb2_grpc.WrapStub(self._connection)
        self.uuid = None
        Local._objects.append(self)

    def import_module(self, package, as_name=None, item=None):
        self._stub.SendImport(ImportRequest(package=package, as_name=as_name, item=item))

    def register_function(self, name):
        self._functions.add(name)

    def _decode_response(self, response: CommandResponse):
        if response.direct:
            return pickle.loads(response.result)

        uuid = pickle.loads(response.result)
        addr = self._address
        if not Local._uuid_lookup.get(addr):
            Local._uuid_lookup[addr] = {}
        if uuid in Local._uuid_lookup[addr]:
            return Local._uuid_lookup[addr][uuid]
        new_local = Local(addr, IndirectLocal)
        new_local.uuid = uuid
        Local._uuid_lookup[addr][uuid] = new_local
        return new_local


    def _interceptor(self, action, *args, **kwargs):
        if not self.uuid:
            raise Exception("Object not instantiated")

        if action == "__getattribute__":
            try:
                known_callable = args[0] in self._functions
                known_name = args[0] if known_callable else None
                item = object.__getattribute__(self, *args) if not known_callable else None
                if known_callable or isinstance(item, Callable):
                    # print(f"call [{known_name or item.__name__}]:, args={args}, kwargs={kwargs}")
                    return lambda *arguments, **keywords: self._decode_response(
                        self._stub.SendCommand(
                            CommandRequest(
                                uuid=self.uuid,
                                method=known_name or item.__name__,
                                payload=pickle.dumps(self._adjust_for_nonlocal(arguments, keywords)),
                            ),
                        )
                    )
            except AttributeError:
                pass

        # print(f"prop [{action}]:, args={args}, kwargs={kwargs}")
        return self._decode_response(
            self._stub.SendCommand(
                CommandRequest(
                    uuid=self.uuid,
                    method=action,
                    payload=pickle.dumps(self._adjust_for_nonlocal(args, kwargs)),
                ),
            )
        )

    def instantiate(self, *arguments, **keywords):
        if self.uuid:
            return
        adjusted_args, adjusted_kwargs, lookups = self._adjust_for_nonlocal(arguments, keywords)
        response: InstanceResponse = self._stub.SendInstance(
            InstanceRequest(
                name=self._cls.__name__,
                payload=pickle.dumps((adjusted_args, adjusted_kwargs, lookups)),
            )
        )
        self.uuid = response.uuid
        Local._uuid_lookup[self.uuid] = self

    @staticmethod
    def _adjust_for_nonlocal(arguments, keywords):
        adjusted_args = []
        adjusted_kwargs = {}
        lookups = []
        for i, arg in enumerate(arguments):
            if isinstance(arg, Local):
                adjusted_args.append(arg.uuid)
                lookups.append(i)
            else:
                adjusted_args.append(arg)
        for i, kwarg in enumerate(keywords):
            value = keywords[kwarg]
            if isinstance(value, Local):
                adjusted_kwargs[kwarg] = value.uuid
                lookups.append(kwarg)
            else:
                adjusted_kwargs[kwarg] = value
        # print(adjusted_args, adjusted_kwargs, lookups)
        return adjusted_args, adjusted_kwargs, lookups

    def __getattribute__(self, name):
        if name in Local._internal_attrs:
            return object.__getattribute__(self, name)
        return self._special_function("__getattribute__", name)

    def __getattr__(self, item):
        return self._special_function("__getattribute__", item)

    def __setattr__(self, name, value):
        if name in Local._internal_attrs:
            object.__setattr__(self, name, value)
        elif hasattr(self, "_special_function"):
            self._special_function("__setattr__", name, value)
        else:
            object.__setattr__(self, name, value)

    def __call__(self, *arguments, **keywords):
        return self._special_function("__call__", *arguments, **keywords)

    @classmethod
    def establish_connection(cls, address) -> grpc.Channel:
        if address in cls._connections:
            return cls._connections[address]
        else:
            cls._connections[address] = grpc.insecure_channel(
                address,
                options=[
                    ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
                    ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
                ],
            )
            return cls._connections[address]


def distributed(original_class, addr, *args, **kwargs):
    return Local(addr, original_class, *args, **kwargs)


class Remote(Generic[T], rwrap_pb2_grpc.WrapServicer):
    def __init__(self, port):
        self.id = 0
        self.objects = {}
        self.port = port

    def SendInstance(self, request: InstanceRequest, context):
        # print("SendInstance", request)
        self.id += 1
        args, kwargs = self._adjust_for_nonlocal(request)
        self.objects[self.id] = globals()[request.name](*args, **kwargs)
        return InstanceResponse(uuid=self.id)

    def _adjust_for_nonlocal(self, request):
        args, kwargs, lookups = pickle.loads(request.payload)
        for lookup in lookups:
            if isinstance(lookup, int):
                args[lookup] = self.objects[args[lookup]]
            else:
                kwargs[lookup] = self.objects[kwargs[lookup]]
        return args, kwargs

    def SendCommand(self, request: CommandRequest, context):
        # print("Command request:", request)
        # print("Payload:", pickle.loads(request.payload))
        # print("Got command...")
        obj = self.objects[request.uuid]
        args, kwargs = self._adjust_for_nonlocal(request)
        f = getattr(obj, request.method)
        result = f(*args, **kwargs)
        try:
            pickled = pickle.dumps(result)
            # print("...returning a direct result")
            return CommandResponse(result=pickled, direct=True)
        except Exception:
            # print("...returning an indirect result")
            self.id += 1
            self.objects[self.id] = result
            return CommandResponse(result=pickle.dumps(self.id), direct=False)

    def SendImport(self, request: ImportRequest, context):
        # print("Import request:", request)
        package = importlib.import_module(request.package)
        if request.item:
            package = getattr(package, request.item)
        globals()[request.as_name or request.item or request.package] = package
        return ImportResponse()

    def SendCleanup(self, request, context):
        # print("Cleanup request:", request)
        self.objects.clear()
        self.id = 0
        return CleanupResponse()

    def start(self):
        print("Starting")
        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=1),
            options=[
                ("grpc.max_send_message_length", MAX_MESSAGE_LENGTH),
                ("grpc.max_receive_message_length", MAX_MESSAGE_LENGTH),
            ],
        )
        rwrap_pb2_grpc.add_WrapServicer_to_server(self, server)
        server.add_insecure_port(f"[::]:{self.port}")
        server.start()
        server.wait_for_termination()
