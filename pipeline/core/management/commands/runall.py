from pipeline.core.management.base import BaseCommand
from pipeline.train import train
from pipeline.predict import predict


class Command(BaseCommand):
    help = (
        "Run all processes to predict test set."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            '-n', '--name',
            help="Use this name for nothing. Just kidding."
        )

    def execute(self, *args, **options):
        train()
        predict()
