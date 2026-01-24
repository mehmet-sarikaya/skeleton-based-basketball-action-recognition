#!/bin/bash
#SBATCH --partition=dev_gpu_a100_il       # Wähle die Partition mit GPUs
#SBATCH --gres=gpu:1            # Fordere 2 GPUs an
#SBATCH --time=48:00:00         # Setze die maximale Laufzeit (hier 2 Stunden)
#SBATCH --mem=32G              # Setze den benötigten Speicher
#SBATCH --cpus-per-task=8      # Anzahl CPUs pro Task
#SBATCH --job-name=device_combis          # Setze den Job-Namen
#SBATCH --output=%x_%j.out     # Datei für die Ausgabe (mit Job-ID)

export MODULEPATH=/opt/bwhpc/common/modulefiles/Core:$MODULEPATH

# Lade die gleichen Module wie im JupyterHub
module load jupyter/ai

# Setze das Arbeitsverzeichnis auf dein Hauptverzeichnis
cd /pfs/data5/home/hr/hr_hr/hr_sarikaym/Basketball_Vision

# Debugging: Zeige das aktuelle Arbeitsverzeichnis
echo "Current working directory: $(pwd)"

# Überprüfe, ob das Python-Skript existiert
ls -l BasketballGestureRecognitionApp.py

# Setze den PYTHONPATH (falls benötigt)
export PYTHONPATH=/pfs/data5/home/hr/hr_hr/hr_sarikaym/Basketball_Vision:$PYTHONPATH


srun python3 BasketballGestureRecognitionApp.py