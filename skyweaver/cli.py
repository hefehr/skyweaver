import argparse
import pprint
import skyweaver

"""
What kind of cli do we want:
app: skyweaver
alias: sw

sw metadata show <metafile>
sw metadata show --key=<key> <metafile>
sw delays create --outfile <outfile> <metafile> <beamformer_config>
sw delays show <delayfile>
sw beamformer run --log-file=<logfile> --log-level=<LEVEL> <delayfile> <metafile> <dataconfig>
sw beamformer monitor <>

We need to be able to do checkpointed recovery somehow aswell. This can probably be 
done per input file.
"""

def metadata_show(args):
    om = skyweaver.ObservationMetadata.from_file(args.metafile)
    if args.verbose:
        pprint.pprint(om)
    else:
        print(om)

def cli():
    parser = argparse.ArgumentParser(prog="skyweaver")
    l1subparsers = parser.add_subparsers(help="sub-command help")
    # sw metadata
    metadata = l1subparsers.add_parser("metadata", help="Tools for observation metadata files")
    metadata_subparsers = metadata.add_subparsers(help="sub-command help")
    
    # sw metadata show
    metadata_show_parser = metadata_subparsers.add_parser("show", help="Show metadata")
    metadata_show_parser.add_argument("--verbose", action="store_true")
    metadata_show_parser.add_argument("metafile", metavar="FILE")
    metadata_show_parser.set_defaults(func=metadata_show)
    args = parser.parse_args()
    args.func(args)
    
  
if __name__ == "__main__":
    cli()
    