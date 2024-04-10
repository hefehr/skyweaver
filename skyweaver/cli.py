import click
import skyweaver

@click.group()
def cli():
    pass

@click.command()
@click.option("--summarize", is_flag=True, 
              help="Summarise the contents of a metafile")
@click.argument("metadata_file")
def metadata(metadata_file, summarize):
    om = skyweaver.ObservationMetadata.from_file(metadata_file)
    if summarize:
        click.echo(om)

cli.add_command(metadata)
  
if __name__ == "__main__":
    cli()
    