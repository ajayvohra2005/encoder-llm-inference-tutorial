import argparse
import torch_xla.core.xla_builder as xb
import pprint

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
                    prog='Convert HLO to text',
                    description='HLO to Text converter',
                    epilog='Text at the bottom of help')
    
    parser.add_argument("--hlo-file",  type=str, help="HLO file path", default="model.hlo.pb")
    args = parser.parse_args()

    with open(args.hlo_file, mode="rb") as f:
        comp = xb.computation_from_module_proto("hlo_pb", f.read())
    
        pprint.pprint(xb.get_computation_hlo(comp))