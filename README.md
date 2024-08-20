# Hydrologic Model-Agnostic Calibration Framework

This repository contains scripts designed to develop a hydrologic model-agnostic calibration framework. The framework supports the calibration of four distinct hydrologic models: **GR4J**, **HYPE**, **MESH**, and **SUMMA**. The calibration process is driven by the **Ostrich** optimization software toolkit.

## Overview

The objective of this framework is to provide a flexible and consistent approach to calibrating different hydrologic models, enabling researchers to conduct model calibration under a unified calibration setup. 

### Models Supported

- **GR4J** (Modèle du Génie Rural à 4 paramètres Journalier): A concpetual rainfall-runoff model widely used for hydrological studies, known for its simplicity and effectiveness in diverse catchments.
- **HYPE** (Hydrological Predictions for the Environment): A conceptual hydrological model developed by the Swedish Meteorological and Hydrological Institute (SMHI).
- **MESH** (Modélisation Environmentale communautaire - Surface Hydrology): A community model for simulating land surface and hydrological processes.
- **SUMMA** (Structure for Unifying Multiple Modeling Alternatives): A flexible modeling framework that allows users to configure different modeling options for hydrologic processes.

### Calibration Tool

- **Ostrich Optimization Software Toolkit:** Ostrich is an open-source tool designed for the calibration of environmental models. It supports various optimization algorithms and is highly customizable, making it an ideal choice for this framework.

## Repository Structure

```plaintext
├── GR4J/      # Scripts required to set up the calibration for GR4J.
├── HYPE/      # Scripts required to set up the calibration for HYPE.
├── MESH/      # Scripts required to set up the calibration for MESH.
├── SUMMA/     # Scripts required to set up the calibration for SUMMA.
├── scripts/   # Scripts required to set up the model-agnostic calibration framework.
├── docs/      # Documentation and guidelines for setting up the model-agnostic calibration framework.
├── examples/  # Example datasets and calibration runs for each hydrologic model to help users get started.
