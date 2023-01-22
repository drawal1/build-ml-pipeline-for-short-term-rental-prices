#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply some basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import wandb
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    artifact_local_path = run.use_artifact(args.input_artifact).file()
    df = pd.read_csv(artifact_local_path)

    logger.log(logging.INFO, "loaded sample.csv artifact into pandas dataframe")

    # Drop outliers
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    logger.info("Keep prices between %f and %f only", min_price, max_price)

    # Convert last_review to datetime
    df['last_review'] = pd.to_datetime(df['last_review'])

    logger.info("converted last review column type to datetime")

    df.to_csv(args.output_artifact, index=False)

    artifact = wandb.Artifact(
        args.output_artifact,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(args.output_artifact)
    run.log_artifact(artifact)

    logger.info("loggging the artifact into wandb")

    run.finish()

    logger.info("finished wandb run")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A very basic data cleaning")

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="name:tag of input artifact (sample.csv:latest)",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="name of output artifact (clean_sample:latest)",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="output type is file",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help = """
                sample data with price outliers removed and last_review column type"
                changed to datetime
               """,
        required=True
    )

    parser.add_argument(
        "--min_price",
        type=float,
        help="minimum daily rate for NYC rentals",
        required=True
    )

    parser.add_argument(
        "--max_price",
        type=float,
        help="maximum daily rate for NYC rentals",
        required=True
    )


    args = parser.parse_args()

    go(args)
