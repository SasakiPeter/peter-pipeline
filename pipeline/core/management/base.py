import os
from argparse import ArgumentParser


class BaseCommand:
    help = ''

    def create_parser(self, prog_name, subcommand):
        parser = ArgumentParser(
            prog=f"{os.path.basename(prog_name)} {subcommand}",
            description=self.help or None,
        )
        self.add_arguments(parser)
        return parser

    def add_arguments(self, parser):
        pass

    def print_help(self, prog_name, subcommand):
        parser = self.create_parser(prog_name, subcommand)
        parser.print_help()

    def run_from_argv(self, argv):
        parser = self.create_parser(argv[0], argv[1])
        options = parser.parse_args(argv[2:])
        cmd_options = vars(options)
        args = cmd_options.pop('args', ())
        # handle_default_options(options)

        self.execute(*args, **cmd_options)

    def execute(self, *args, **options):
        raise NotImplementedError(
            'subclasses of BaseCommand must provide a execute() method')
