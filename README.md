# Thesis Modeling and Computational Biology Demos

This repository contains a curated set of simplified computational demos inspired by the methodological framework of my master’s thesis in Medical Biotechnologies at the Università del Piemonte Orientale.

My thesis focuses on mathematical modelling and sensitivity analysis of erythropoiesis disruption in acute myeloid leukaemia (AML), with particular attention to nurse macrophages, erythroblastic islands, and niche-mediated regulation. The broader project combines ordinary differential equation (ODE) modelling, simulation, calibration, and sensitivity analysis to study how leukaemia-driven microenvironmental changes alter erythroid dynamics and to explore intervention-oriented hypotheses. :contentReference[oaicite:1]{index=1}

## Repository scope

At present, this repository is intentionally limited to **generic, educational, and public-safe demo scripts** that reflect the main computational ideas used throughout the thesis, including:

- ODE-based dynamic modelling
- parameter estimation and calibration workflows
- local and global sensitivity analysis
- intervention-response simulations
- surrogate and uncertainty-analysis style demonstrations

These scripts are designed to communicate the computational style and methodological direction of the work without prematurely disclosing unpublished biological details, calibrated parameter sets, manuscript-specific analyses, or publication-linked research code.

## Why the repository currently contains demos

Parts of the thesis project are connected to ongoing research outputs and material that is better released in a more complete and publication-linked form. For that reason, the current repository does **not** contain the full research codebase used in the thesis or manuscript-related analyses.

Instead, the repository provides representative demo implementations of the types of models and workflows developed during the project. The goal is to make the computational approach visible in a responsible and structured way while preserving appropriate separation between public educational material and unpublished research assets.

## Thesis context

The thesis integrates experimental and computational work to investigate how AML remodels the bone marrow microenvironment and perturbs erythropoiesis. It includes a mechanistic modelling framework, model calibration against experimental observations, and both local and global sensitivity analyses to identify influential regulatory processes and evaluate potential intervention strategies. The thesis also includes benchmark toy models, ODE-based computational implementation, and analysis of intervention concepts such as cytokine-modulation scenarios. :contentReference[oaicite:2]{index=2}

The demos in this repository are therefore **conceptually related** to the thesis, but they should not be interpreted as the final archived or publication-ready research code.

## What the current scripts are intended to show

The files currently included here are meant to demonstrate:

- how dynamic biological systems can be represented using compartmental ODE models
- how synthetic or simplified calibration workflows can be structured
- how sensitivity-analysis concepts can be implemented in practice
- how intervention-response logic can be explored computationally in reduced toy settings
- how modelling workflows can be written in a clear and reproducible way

In other words, this repository is currently a **public-facing methodological companion**, not the complete scientific software release associated with the thesis.

## Planned future updates

Following publication or formal release of the associated research outputs, this repository may be expanded to include:

- more complete research-oriented code
- publication-linked reproducible workflows
- clearer mapping between the public demos and the final scientific analyses
- additional documentation for the full modelling framework
- more detailed notes on parameterisation, validation, and sensitivity-analysis strategy

## Important note

The current scripts should be understood as **representative demonstrations** only. They are intentionally simplified and abstracted, and they omit thesis-specific biological detail, unpublished data dependencies, fitted model components, and manuscript-level analyses.

## Author

**Shayan Jalali**  
MSc in Medical Biotechnologies  
Università del Piemonte Orientale  
Computational biology, mathematical modelling, and data analysis
