def read_file(path: str) -> str:
    """Read a text file and return its content as a string."""
    with open(path, "r") as f:
        return f.read()


def write_csv(df, path: str) -> None:
    """Write a DataFrame to CSV."""
    df.to_csv(path, index=False)


def write_json(metadata, df_kernels, df_regions, path: str) -> None:
    """Write DataFrames to JSON."""
    import json

    output = {
        "metadata": metadata,
        "kernels": df_kernels.to_dict(orient="records"),
        "regions": df_regions.to_dict(orient="records"),
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=4)
