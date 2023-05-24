# main.py
from lightning.pytorch.cli import LightningCLI

# simple demo classes for your convenience
import .datamodule, .module

def cli_main():
    cli = LightningCLI(DemoModel, BoringDataModule)
    # note: don't call fit!!


if __name__ == "__main__":
    cli_main()
    # note: it is good practice to implement the CLI in a function and call it in the main if block