import argparse, json, os
from datetime import datetime

DATASET_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
ARGS_DATE_FORMAT = '%Y-%m-%d'

parser = argparse.ArgumentParser(description='Splits a dataset into others using years as delimiter.')

parser.add_argument('--dataset', type=str, help='dataset path. A JSON file', required=True)
parser.add_argument('--outputPath', type=str, help='path to put the resulting split datasets', required=True)
parser.add_argument('--years', nargs='+', help='years to use as delimiters while splitting', required=True)

args = parser.parse_args()

original_dataset = json.load(open(args.dataset, 'r'))

dataset_name = args.dataset.split('/')[-1]

for year_string in args.years:
    year = datetime.strptime(year_string, ARGS_DATE_FORMAT)

    year_dataset = list(filter(lambda record: datetime.strptime(record['date'], DATASET_DATE_FORMAT) >= year, original_dataset))

    path = f'{args.outputPath}/{dataset_name}_[{year_string}_onwards_dataset].json'

    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(year_dataset, open(path, 'w'))

path = f'{args.outputPath}/{dataset_name}_[original_dataset].json'
os.makedirs(os.path.dirname(path), exist_ok=True)
json.dump(original_dataset, open(path, 'w'))

print(f'Datasets saved to {args.outputPath}/ folder.')
