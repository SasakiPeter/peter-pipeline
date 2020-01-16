from pipeline.core.management.base import BaseCommand
from pipeline.train import train


class Command(BaseCommand):
    help = (
        "Train models automatically."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '-n', '--name',
            help="Use this name for nothing. Just kidding."
        )

    def execute(self, *args, **options):
        train()
