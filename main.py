from control.config_parameters import get_args, get_config
from control.Server import Server
from control.MMULFED.ServerMMULFED import ServerMMULFED
from control.Enums import FLFramework


if __name__ == "__main__":

    args = get_args()
    print((f"Arguments:", dict(args._get_kwargs())))
    args = get_config(args)

    server = None
    if args.fl_framework in (FLFramework.MMULFED.value, FLFramework.MMULFEDAA.value):
        "start creating an MMULFED server"
        server = ServerMMULFED(args)
    elif args.fl_framework == FLFramework.FEDAVG.value:
        "start creating a server"
        server = Server(args)
    else:
        assert False
    server.run()
