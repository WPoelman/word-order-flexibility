import argparse
import ssl
import tarfile
import zipfile
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def safe_extract_member(tar: tarfile.TarFile, member: tarfile.TarInfo, target: Path) -> None:
    """Extract one archive member without allowing path traversal."""
    root = target.resolve()
    destination = (target / member.name).resolve()
    if not destination.is_relative_to(root):
        raise ValueError(f"Archive member escapes the target directory: {member.name}")
    if member.issym() or member.islnk():
        raise ValueError(f"Archive links are not supported: {member.name}")
    tar.extract(member, path=target)


def download_and_extract_ud(target_dir):
    """Download and extract Universal Dependencies treebanks."""
    url = "https://lindat.mff.cuni.cz/repository/server/api/core/items/b4fcb1e0-f4b2-4939-80f5-baeafda9e5c0/allzip?handleId=11234/1-6036"

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    # The URL returns a zip file, not tar.gz like listed on the website
    download_file = target_dir / "ud-treebanks-v2.17.zip"

    print("Universal Dependencies 2.17")
    print(f"URL: {url}")
    print(f"Target: {target_dir}\n")

    extracted_dir = target_dir / "ud-treebanks-v2.17"
    if extracted_dir.exists() and any(extracted_dir.iterdir()):
        print(f"Already extracted to {extracted_dir}")
        print("To re-download, delete the directory and run again.")
        return

    if download_file.exists():
        print(f"Archive already downloaded: {download_file}")
    else:
        print(f"Downloading from {url} this is about 800mb so it can take a while...")
        try:
            # lindat.mff.cuni.cz uses a certificate chain that the VSC HPC
            # doesn't trust, so we fall back to an unverified context
            # when the default one fails.
            try:
                resp = urlopen(url)
            except (ssl.SSLCertVerificationError, URLError):
                ctx = ssl.create_default_context()
                ctx.check_hostname = False
                ctx.verify_mode = ssl.CERT_NONE
                resp = urlopen(url, context=ctx)
            with open(download_file, "wb") as f:
                while chunk := resp.read(1024 * 1024):
                    f.write(chunk)
            print(f"Downloaded to {download_file}")
        except Exception as error:
            raise RuntimeError("Failed to download Universal Dependencies") from error

    print("\nExtracting archive (treebanks only, skipping tools and docs)...")
    try:
        treebanks_tgz = target_dir / "ud-treebanks-v2.17.tgz"

        with zipfile.ZipFile(download_file, "r") as zf:
            zf.extract("ud-treebanks-v2.17.tgz", path=target_dir)
            print("Extracted treebanks tarball from zip")

        print("Unpacking treebanks tarball...")
        with tarfile.open(treebanks_tgz, "r:gz") as tar:
            members = tar.getmembers()
            total = len(members)

            for i, member in enumerate(members, 1):
                safe_extract_member(tar, member, target_dir)
                if i % 100 == 0 or i == total:
                    percent = (i / total) * 100
                    print(
                        f"\rExtracting: {percent:.1f}% ({i}/{total} files)",
                        end="",
                        flush=True,
                    )

        print(f"\nExtracted to {extracted_dir}")
        print("Cleaning up...")
        download_file.unlink()
        treebanks_tgz.unlink()
        print("Removed temporary files")

    except Exception as error:
        raise RuntimeError("Failed to extract Universal Dependencies") from error

    print("\nVerifying...")
    treebank_dirs = list(extracted_dir.glob("UD_*"))
    print(f"Found {len(treebank_dirs)} treebanks")

    print("Download complete")


def main():
    parser = argparse.ArgumentParser(
        description="Download and extract Universal Dependencies treebanks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download UD 2.17 to default location
  python download_ud.py

  # Download to custom location
  python download_ud.py --target /path/to/datasets
        """,
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path(__file__).parent / "ud",
        help="Target directory for treebanks.",
    )

    args = parser.parse_args()
    args.target.mkdir(exist_ok=True, parents=True)

    download_and_extract_ud(target_dir=args.target)


if __name__ == "__main__":
    main()
