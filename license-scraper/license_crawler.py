import difflib
import logging
import os
from argparse import ArgumentParser
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd

from finder import GithubLinkFinder
from scraper import GithubScraper, HuggingFaceScraper, TFHubScraper


def main(args):
    if args.url:
        print(find_license(args.url))
        return

    if not args.url:
        csv_path = ""

        if args.csv and not args.google_sheet:
            csv_path = args.csv

        if args.google_sheet and not args.csv:
            sheet_id, sheet_name = args.google_sheet.split(":")
            logging.debug(f"sheet_id:{sheet_id}, sheet_name:{sheet_name}")
            url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
            csv_path = url

        df = pd.read_csv(csv_path)
        df['License'] = parallelize_on_rows(df[df['Accessible'].notna()]['Accessible'], find_license)

        if args.clean:
            df = tidy_up_licenses(df)

        if args.print == "CONSOLE":
            print(df[['Name', 'License']])
        if args.print == "CSV":
            if not os.path.exists(args.output_dir):
                os.mkdir(args.output_dir)
            csv_output_path = os.path.join(args.output_dir, "license_output.csv")
            df[['Name', 'License']].to_csv(csv_output_path)
            logging.info(f"Licenses saved to {csv_output_path}")


def find_license(url):
    if "huggingface.co" in url:
        return HuggingFaceScraper().scrape_license(url)
    if "tfhub.dev" in url:
        return TFHubScraper().scrape_license(url)

    return GithubScraper().scrape_license(
        GithubLinkFinder().find_from_url(url)
    )


def get_words_list(df):
    df['License'] = df['License'].apply(lambda x: x.lower().replace("license", "").strip(" ") if type(x) is str else x)

    words_list = [x[:10] for x in df['License'].value_counts().keys().to_list()]
    new_word_list = []
    while len(words_list) != 0:
        word = words_list[0]

        words_list.remove(word)
        matches = difflib.get_close_matches(word, words_list)

        words_list = [word for word in words_list if word not in matches]
        new_word_list.append(word)

    new_word_list = [difflib.get_close_matches(word, df['License'].value_counts().keys().to_list())[0] for
                     word in new_word_list]
    logging.debug(f"Licenses found: {new_word_list}")
    return new_word_list


def tidy_up_licenses(df):
    word_list = get_words_list(df)

    def clean(x):
        close = difflib.get_close_matches(str(x), word_list)
        if len(close) < 1:
            return x
        if str(x) != close[0]:
            logging.debug(f"Transform {str(x)} to {close[0]}")
        return close[0]

    df['License'] = df['License'].apply(clean)
    return df


def parallelize(data, func, num_of_processes=8):
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes)
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def run_on_subset(func, data_subset):
    return data_subset.apply(func)


def parallelize_on_rows(data, func, num_of_processes=8):
    return parallelize(data, partial(run_on_subset, func), num_of_processes)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--info", action="store_true")

    parser.add_argument("--url", type=str)
    parser.add_argument("--csv", type=str)
    parser.add_argument("--google_sheet", type=str,
                        help="Format of google sheet has to be like: {sheet-id}:{sheet-name}")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("-l", "--log", dest="logLevel", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help="Set the logging level")
    parser.add_argument("-o", "--output_dir", type=str, default="output/")
    parser.add_argument("-p", "--print", type=str, choices=['CONSOLE', 'CSV'], default="CSV")
    parser.add_argument("-c", "--clean", action="store_true")

    _args = parser.parse_args()

    if _args.logLevel:
        logging.basicConfig(level=getattr(logging, _args.logLevel))

    main(_args)
