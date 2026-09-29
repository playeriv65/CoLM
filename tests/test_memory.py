"""Peak memory of every rank: aggregation of the per-rank, per-phase peaks."""

from colm.train.memory import MemoryMeter


def _rank(select, train, reserved=1.0):
    return {
        "selection": {"allocated": select, "reserved": select + reserved},
        "train": {"allocated": train, "reserved": train + reserved},
    }


def test_summary_reports_the_maximum_and_every_rank():
    ranks = [_rank(3.0, 20.0), _rank(2.0, 24.0), _rank(2.5, 21.0)]
    summary = MemoryMeter.summary(ranks)
    assert summary["peak_mem_gb"] == 24.0 and summary["peak_reserved_gb"] == 25.0
    assert summary["peak_mem_selection_gb"] == 3.0 and summary["peak_mem_train_gb"] == 24.0
    assert [summary[f"peak_mem_gb_rank{i}"] for i in range(3)] == [20.0, 24.0, 21.0]


def test_single_rank_and_cpu_runs():
    assert "peak_mem_gb_rank0" not in MemoryMeter.summary([_rank(1.0, 2.0)])
    assert MemoryMeter.summary([{}]) == {}  # no GPU: nothing was recorded


def test_the_meter_gathers_and_clears_its_window():
    meter = MemoryMeter()
    meter.window = {"train": {"allocated": 1.0, "reserved": 2.0}}
    meter.run = {"train": {"allocated": 5.0, "reserved": 6.0}}
    assert meter.gather() == [{"train": {"allocated": 1.0, "reserved": 2.0}}] and meter.window == {}
    assert meter.gather(run=True)[0]["train"]["allocated"] == 5.0
