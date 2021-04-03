import pandas as pd
import json
import os
import argparse
import numpy as np


def main(opts):
    with open(os.path.join(opts.input_folder, opts.pred_file), "r") as f:
        data = json.load(f)
        probs_grp = []
        ids_grp = []
        ordered_data = sorted(data.items(),
                              key=lambda item: int(item[0].split("-")[1]))
        for annot_id, scores in ordered_data:
            ids_grp.append(annot_id)
            probs_grp.append(np.array(scores).reshape(1, 5, 4))

    # Double check the IDs are in the same order for everything
    # assert [x == ids_grp[0] for x in ids_grp]

    probs_grp = np.stack(probs_grp, 1)
    # essentially probs_grp is a [num_ex, 5, 4] array of probabilities.
    # The 5 'groups' are
    # [answer, rationale_conditioned_on_a0, rationale_conditioned_on_a1,
    #          rationale_conditioned_on_a2, rationale_conditioned_on_a3].
    # We will flatten this to a CSV file so it's easy to submit.
    group_names = ['answer'] + [f'rationale_conditioned_on_a{i}' 
                                for i in range(4)]
    probs_df = pd.DataFrame(data=probs_grp.reshape((-1, 20)),
                            columns=[f'{group_name}_{i}'
                            for group_name in group_names for i in range(4)])
    probs_df['annot_id'] = ids_grp
    probs_df = probs_df.set_index('annot_id', drop=True)
    probs_df.to_csv(os.path.join(opts.input_folder, opts.output_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--pred_file",
                        default=None, type=str,
                        help="The input JSON file.")
    parser.add_argument("--output_file",
                        default=None, type=str,
                        help="The output CSV file.")
    parser.add_argument(
        "--input_folder", default=None, type=str,
        help="The directory where the predicted JSON files are in")

    args = parser.parse_args()

    main(args)

