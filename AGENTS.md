# Uncertainty Guided Agent Reasoning Project

This repository contains experements for university thesis. It explores how different uncernatity scores, influence software agent reasoning process

## Used Uncertanity Estimation Methods

Evaluated methods are the following:

* Lookback Lens - https://arxiv.org/pdf/2407.07071
* Uncertainty Quantification Heads - https://arxiv.org/pdf/2505.08200v1
* ICR Probe - https://arxiv.org/pdf/2507.16488
* SAPLMA - https://arxiv.org/pdf/2304.13734
* HaloScope - https://arxiv.org/pdf/2409.17504

Agent loop implemented in src/agent_loop.py. General pipeline is the following:
1. Train estimator
2. Use estimator in agent loop

    ```mermaid
    flowchart TD
        input["Task and archived draft"] --> inspect["Inspect current draft"]
        inspect --> mode{"Uncertainty estimator enabled?"}
        mode -->|Yes| rank["Rank lines with SAPLMA or UQ Head"]
        mode -->|No| review["Review and propose a revision"]
        rank -->|Top two lines| review
        review --> valid{"Complete, nonempty code block?"}
        valid -->|No| keep["Keep current draft"]
        valid -->|Yes| stop{"Draft unchanged or round limit reached?"}
        stop -->|No| inspect
        stop -->|Yes| picked["Return revised draft"]
        keep --> grade["Benchmark evaluation"]
        picked --> grade
        reference["Canonical code and available Python assertions"] --> grade
    ```

## Code style

* Follow PEP8 style. Tab = 4 spaces, use typing
* No need in comments or docstrings, unless user states it

## Agents MUST

1. User is stopid respond coherently in simple, clear words.
2. Keep the project clean.
3. Ask the user before you make a new file, copy file, remove file or perform any other change that will significantly change projects structure.
4. If a new feature needs changes to old files - copy the old files and change only the copies.
5. When you add a new feature, make smoke tests for it. Run them. Check that the feature works and the tests pass. Also new feature should not break already existing tests.
6. You can freely make new tests, but you can change a test only if the user approves.
7. When running code, always use `.venv` python venv environment.
