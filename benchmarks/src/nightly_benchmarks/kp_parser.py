import re
import pandas as pd
from datetime import datetime
from typing import Dict, Any


def parse_regions_section(text: str) -> pd.DataFrame:
    """
    Parse the Regions section of Kokkos profiler output into a pandas DataFrame.
    """
    lines = text.splitlines()
    regions = []
    i = 0

    # Find the start of Regions section
    while i < len(lines):
        if lines[i].strip() == "Regions:":
            i += 2  # Skip "Regions:" and empty line
            break
        i += 1

    # Parse regions until we hit "Kernels:" section or end
    while i < len(lines):
        line = lines[i].strip()

        # Stop if we hit the Kernels section
        if line.startswith("Kernels:") or line.startswith("-------------------------"):
            break

        # Region name line
        if line.startswith("- "):
            region_name = line[2:].strip()

            # Next line contains numbers
            if i + 1 < len(lines):
                match = re.match(
                    r"\((\w+)\)\s+([\d\.]+)\s+(\d+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)",
                    lines[i + 1].strip(),
                )
                if match:
                    exec_type, time, calls, avg, p1, p2 = match.groups()
                    regions.append(
                        {
                            "region": region_name,
                            "exec_type": exec_type,
                            "time": float(time),
                            "calls": int(calls),
                            "avg_time": float(avg),
                            "percent1": float(p1),
                            "percent2": float(p2),
                        }
                    )
                i += 1  # skip numeric line
        i += 1

    return pd.DataFrame(regions)


def parse_kernels_section(text: str) -> pd.DataFrame:
    """
    Parse the Kernels section of Kokkos profiler output.
    Returns a DataFrame with detailed kernel timing information.
    """
    lines = text.splitlines()
    kernels = []
    i = 0

    # Find the start of Kernels section
    while i < len(lines):
        if lines[i].strip() == "Kernels:":
            i += 2  # Skip "Kernels:" and empty line
            break
        i += 1

    # Parse kernels until we hit summary section or end
    while i < len(lines):
        line = lines[i].strip()

        # Stop if we hit the summary section or another major section
        if (
            line.startswith("-------------------------")
            or line.startswith("Total Execution Time")
            or line.startswith("# ")
            or line == ""
        ):
            # Check if this is just a separator within kernels section
            if line.startswith("-------------------------"):
                # Look ahead to see if there's more kernel data
                j = i + 1
                found_more_kernels = False
                while j < len(lines) and j < i + 5:  # Look ahead a few lines
                    next_line = lines[j].strip()
                    if (
                        next_line
                        and not next_line.startswith("#")
                        and "(" in next_line
                        and ")" in next_line
                    ):
                        found_more_kernels = True
                        break
                    j += 1

                if not found_more_kernels:
                    break

        # Parse kernel lines: "KERNEL_NAME(ARGS) COUNT TIME_PER_CALL TOTAL_TIME PERCENT"
        if line and not line.startswith("#"):
            # Use regex to parse kernel lines more robustly
            kernel_pattern = r"^(.+?)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)%?\s*$"
            match = re.match(kernel_pattern, line)

            if match:
                kernel_name = match.group(1).strip()
                count = int(match.group(2))
                time_per_call = float(match.group(3))
                total_time = float(match.group(4))
                percent = float(match.group(5))

                kernels.append(
                    {
                        "kernel_name": kernel_name,
                        "count": count,
                        "time_per_call": time_per_call,
                        "total_time": total_time,
                        "percent": percent,
                    }
                )

        i += 1

    return pd.DataFrame(kernels)


def parse_kokkos_output(text: str) -> pd.DataFrame:
    """
    Parse Kokkos profiler output into a pandas DataFrame.
    """
    lines = text.splitlines()
    kernels = []
    i = 0

    # Find the start of Kernels section
    while i < len(lines):
        if lines[i].strip() == "Kernels:":
            i += 2  # Skip "Kernels:" and empty line
            break
        i += 1

    # Parse kernels until we hit summary section or end
    while i < len(lines):
        line = lines[i].strip()

        # Stop if we hit the summary section
        if line.startswith("-------------------------"):
            break

        # Kernel name line
        if line.startswith("- "):
            kernel_name = line[2:].strip()

            # Next line contains numbers
            if i + 1 < len(lines):
                match = re.match(
                    r"\((\w+)\)\s+([\d\.]+)\s+(\d+)\s+([\d\.]+)\s+([\d\.]+)\s+([\d\.]+)",
                    lines[i + 1].strip(),
                )
                if match:
                    exec_type, time, calls, avg, p1, p2 = match.groups()
                    kernels.append(
                        {
                            "kernel": kernel_name,
                            "exec_type": exec_type,
                            "time": float(time),
                            "calls": int(calls),
                            "avg_time": float(avg),
                            "percent1": float(p1),
                            "percent2": float(p2),
                        }
                    )
                i += 1  # skip numeric line
        i += 1

    return pd.DataFrame(kernels)


def total_execution_time(text: str) -> float:
    """
    Extract total execution time from the summary section of Kokkos profiler output.
    """
    lines = text.splitlines()
    total_time = 0.0

    for line in lines:
        if "Total Execution Time" in line:
            match = re.search(r"Total Execution Time.*?:\s+([\d\.]+)", line)
            if match:
                total_time = float(match.group(1))
                break

    return total_time


def parse_metadata(text: str) -> Dict[str, Any]:
    """
    Extract metadata information including timestamp, benchmark name, and hardware details.
    """
    metadata = {}
    lines = text.splitlines()

    hardware_info = {}

    for line in lines:
        line = line.strip()

        # Extract current time
        if line.startswith("# Current Time"):
            continue
        elif (
            line
            and not line.startswith("#")
            and "EDT" in line
            or "EST" in line
            or "PST" in line
            or "PDT" in line
        ):
            # Parse timestamp line like "Tue Sep 16 03:32:35 PM EDT 2025"
            try:
                # Remove timezone abbreviations for parsing
                cleaned_time = re.sub(r"\s+(EDT|EST|PST|PDT|UTC)\s+", " ", line)
                # Try different datetime formats
                for fmt in ["%a %b %d %I:%M:%S %p %Y", "%a %b %d %H:%M:%S %Y"]:
                    try:
                        parsed_time = datetime.strptime(cleaned_time, fmt)
                        metadata["timestamp"] = parsed_time.isoformat()
                        metadata["timestamp_raw"] = line
                        break
                    except ValueError:
                        continue
            except Exception:
                metadata["timestamp_raw"] = line

        # Extract benchmark name
        elif line.startswith("# Benchmark:"):
            benchmark_name = line.replace("# Benchmark:", "").strip()
            metadata["benchmark_name"] = benchmark_name

        # Extract hardware information
        elif ":" in line and not line.startswith("#"):
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            # Map important hardware fields
            hardware_fields = {
                "Architecture": "architecture",
                "CPU(s)": "cpu_count",
                "Vendor ID": "cpu_vendor",
                "Model name": "cpu_model",
                "CPU family": "cpu_family",
                "Model": "cpu_model_number",
                "Thread(s) per core": "threads_per_core",
                "Core(s) per socket": "cores_per_socket",
                "Socket(s)": "socket_count",
                "CPU max MHz": "cpu_max_mhz",
                "CPU min MHz": "cpu_min_mhz",
                "BogoMIPS": "bogomips",
                "L1d cache": "l1d_cache",
                "L1i cache": "l1i_cache",
                "L2 cache": "l2_cache",
                "L3 cache": "l3_cache",
                "NUMA node(s)": "numa_nodes",
            }

            if key in hardware_fields:
                field_name = hardware_fields[key]
                # Convert numeric fields
                if field_name in [
                    "cpu_count",
                    "cpu_family",
                    "cpu_model_number",
                    "threads_per_core",
                    "cores_per_socket",
                    "socket_count",
                    "numa_nodes",
                ]:
                    try:
                        hardware_info[field_name] = int(value)
                    except ValueError:
                        hardware_info[field_name] = value
                elif field_name in ["cpu_max_mhz", "cpu_min_mhz", "bogomips"]:
                    try:
                        hardware_info[field_name] = float(value)
                    except ValueError:
                        hardware_info[field_name] = value
                else:
                    hardware_info[field_name] = value
    # Calculate derived metrics
    if "cpu_count" in hardware_info and "socket_count" in hardware_info:
        hardware_info["cpus_per_socket"] = (
            hardware_info["cpu_count"] // hardware_info["socket_count"]
        )

    if "cores_per_socket" in hardware_info and "threads_per_core" in hardware_info:
        hardware_info["logical_cores_per_socket"] = (
            hardware_info["cores_per_socket"] * hardware_info["threads_per_core"]
        )

    # Add parsing timestamp
    metadata["hardware"] = hardware_info

    return metadata


def parse_complete_kokkos_output(filepath):
    """Parse Kokkos profiler output file and return all structured data."""
    with open(filepath, "r") as f:
        content = f.read()

    result = {}

    # Parse metadata (timestamp, benchmark name, hardware info)
    metadata = parse_metadata(content)
    if metadata:
        result["metadata"] = metadata

    # Parse regions section (high-level operations)
    regions_df = parse_regions_section(content)
    if not regions_df.empty:
        result["regions"] = regions_df

    # Parse kernels section (detailed kernel operations)
    kernels_df = parse_kernels_section(content)
    if not kernels_df.empty:
        result["kernels"] = kernels_df

    # Calculate total execution time
    total_time = total_execution_time(content)
    if total_time is not None:
        result["total_execution_time"] = total_time

    return result
