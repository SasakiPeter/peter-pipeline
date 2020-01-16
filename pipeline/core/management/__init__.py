
import os
import sys
# import functools
from importlib import import_module
import pkgutil
from argparse import ArgumentParser


def find_commands(management_dir):
    command_dir = os.path.join(management_dir, 'commands')
    return [name for _, name, is_pkg in pkgutil.iter_modules([command_dir])
            if not is_pkg and not name.startswith('_')]


def load_command_class(name):
    module = import_module(f"pipeline.core.management.commands.{name}")
    return module.Command()


def get_commands():
    commands = [name for name in find_commands(__path__[0])]
    return commands


class ManagementUtility:
    def __init__(self, argv=None):
        self.argv = argv or sys.argv[:]
        self.prog_name = os.path.basename(self.argv[0])

    def fetch_command(self, subcommand):
        commands = get_commands()
        if subcommand not in commands:
            raise KeyError
        return load_command_class(subcommand)

    def execute(self):
        try:
            subcommand = self.argv[1]
        except IndexError:
            subcommand = 'help'

        parser = ArgumentParser(
            usage='%(prog)s subcommand [options] [args]',
            add_help=False, allow_abbrev=False)
        parser.add_argument('args', nargs='*')

        try:
            options, args = parser.parse_known_args(self.argv[2:])
        except Exception:
            pass

        if subcommand == 'help':
            # sys.stdout.write(self.main_help_text() + '\n')
            if not options.args:
                print('display all help')
            else:
                self.fetch_command(options.args[0]).print_help(
                    self.prog_name, options.args[0])
        else:
            self.fetch_command(subcommand).run_from_argv(self.argv)


def execute_from_command_line(argv=None):
    utility = ManagementUtility(argv)
    utility.execute()
