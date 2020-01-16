from pipeline.core.management.base import BaseCommand
from pipeline.predict import make_submit_file


class Command(BaseCommand):
    help = (
        "Create submit file."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '-n', '--name',
            help="Use this name for nothing. Just kidding."
        )

    def execute(self, *args, **options):
        make_submit_file()
