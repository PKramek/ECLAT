import argparse
import logging
from time import time

import matplotlib.pyplot as plt

from AssociationRules.association_rules import AssociationRulesGenerator

# Example how to run this program
a = "python main.py -m_conf 0.95 -m_supp 0.001 --path ./data/BMS1_itemset_mining.txt -csv ./results/results_testing.csv"


def graph_metric(results_df, metric: str):
    metrics_color_lookup = {
        'cosine': 'r',
        'certainty_factor': 'blue',
        'conviction': 'g'}

    metric_lookup = {
        'cosine': results_df.cosine,
        'certainty_factor': results_df.certainty_factor,
        'conviction': results_df.conviction}

    y_label_lookup = {
        'cosine': 'Cosine value',
        'certainty_factor': 'Certainty factor value',
        'conviction': 'Conviction value'}

    assert metric in metrics_color_lookup.keys()
    plt.scatter(results_df.lift, metric_lookup[metric], c=metrics_color_lookup[metric], s=5, alpha=0.6)
    plt.xlabel('Lift value')
    plt.ylabel(y_label_lookup[metric])
    plt.grid()
    plt.show()


parser = argparse.ArgumentParser(prog="Eclat", description='Find association rules in data set')
parser.add_argument('-m_conf', '--min_confidence', type=float, required=True,
                    help='Minimal association rule confidence, should be a value in interval [0,1]')
parser.add_argument('-m_supp', '--min_support', type=float, required=True,
                    help='Minimal frequent itemset support, , should be a value in interval [0,1]')
parser.add_argument('-p', '--path', type=str, required=True, help='Path to file containing dataset')
parser.add_argument('--no-cosine', action='store_true', help='Should cosine metric be calculated')
parser.add_argument('--no-conviction', action='store_true', help='Should conviction metric be calculated')
parser.add_argument('--no-certainty_f', action='store_true', help='Should certainty factor metric be calculated')
parser.add_argument('--cosine_graph', action='store_true',
                    help='Should cosine metric be graphed in respect to lift metric')
parser.add_argument('--conviction_graph', action='store_true',
                    help='Should conviction metric be graphed in respect to lift metric')
parser.add_argument('--certainty_f_graph', action='store_true',
                    help='Should certainty factor metric be graphed in respect to lift metric')
parser.add_argument('-csv', '--csv_output', type=str, default=None,
                    help='Path to csv file in which output will be saved')
parser.add_argument('-excel', '--excel_output', type=str, default=None,
                    help='Path to excel file in which output will be saved')
parser.add_argument('-json', '--json_output', type=str, default=None,
                    help='Path to json file in which output will be saved')
parser.add_argument('-print', '--print_to_console', action='store_true',
                    help='Should found rules be printed to console')
parser.add_argument('-log', '--logging', type=str, default='Info', choices=['Info', 'Debug', 'None'],
                    help='Should logs be printed to console, Debug level not recommended while creating graphs.')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.logging != 'None':
        log_format = '%(asctime)s - %(message)s'
        if args.logging == 'Info':
            logging.basicConfig(level=logging.INFO, format=log_format)
        elif args.logging == 'Debug':
            logging.basicConfig(level=logging.DEBUG, format=log_format)
        else:
            raise ValueError('Not known type level of logging.')

    if args.cosine_graph:
        if args.no_cosine:
            raise AttributeError('Can not graph cosine without calculating it first')

    if args.conviction_graph:
        if args.no_conviction:
            raise AttributeError('Can not graph conviction without calculating it first')

    if args.certainty_f_graph:
        if args.no_certainty_f:
            raise AttributeError('Can not graph certainty factor without calculating it first')

    start_time = time()

    association_rules_generator = AssociationRulesGenerator(
        AssociationRulesGenerator.ECLAT, args.path,
        min_support=args.min_support, min_confidence=args.min_confidence,
        cosine=not args.no_cosine, certainty_f=not args.no_certainty_f, conviction=not args.no_conviction)

    results_dataframe = association_rules_generator.find_rules()

    if args.print_to_console:
        association_rules_generator.print_results_to_console()

    if args.csv_output is not None:
        association_rules_generator.save_results_to_csv(args.csv_output)
    if args.excel_output is not None:
        association_rules_generator.save_results_to_excel(args.excel_output)
    if args.json_output is not None:
        association_rules_generator.save_results_to_json(args.json_output)

    if args.conviction_graph:
        graph_metric(results_dataframe, 'conviction')

    if args.cosine_graph:
        graph_metric(results_dataframe, 'cosine')

    if args.certainty_f_graph:
        graph_metric(results_dataframe, 'certainty_factor')

    logging.info('Execution time: {}'.format(time() - start_time))
