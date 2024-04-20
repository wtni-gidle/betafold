from argparse import ArgumentParser


def remove_arguments(parser: ArgumentParser, args):
    for arg in args:
        for action in parser._actions:
            opts = vars(action)["option_strings"]
            if arg in opts:
                parser._handle_conflict_resolve(None, [(arg, action)])
