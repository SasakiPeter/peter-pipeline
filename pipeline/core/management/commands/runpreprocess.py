from pipeline.core.management.base import BaseCommand
from pipeline.preprocess import preprocess


class Command(BaseCommand):
    help = (
        "Run preprocessing."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '-n', '--name',
            help="Use this name for nothing. Just kidding."
        )

    def execute(self, *args, **options):
        preprocess()
