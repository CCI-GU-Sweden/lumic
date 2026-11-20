"""Download the COS-7 spectral imaging examples from BioImage Archive."""

import argparse

from lumic.data import download_cos7_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snr", choices=("high", "medium", "low", "all"), default="high")
    parser.add_argument("--include-references", action="store_true",
                        help="Download the single-stain acquisitions too.")
    args = parser.parse_args()

    download_cos7_data(snr=args.snr, include_references=args.include_references, progress=True)


if __name__ == "__main__":
    main()
