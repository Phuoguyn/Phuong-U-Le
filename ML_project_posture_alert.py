{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPR2hsS6eQGsAotiJ35RLj1",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Phuoguyn/Phuong-U-Le/blob/main/ML_project_posture_alert.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#appDomain\n",
        "\n",
        "#This project is situated in the domain of digital wellness and ergonomic support, focusing on maintaining healthy posture during extended computer use. It helps students develop posture awareness through real-time webcam feedback and gentle corrective prompts."
      ],
      "metadata": {
        "id": "BhFgC_E0dMVf"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#app.py\n",
        "# Typing Posture Alert (Streamlit + MediaPipe + streamlit-webrtc)\n",
        "# Features:\n",
        "#   • Consent gate\n",
        "#   • Webcam posture detection (neck/back/head)\n",
        "#   • Personal calibration (baseline angles)\n",
        "#   • On-screen coaching alerts for bad posture\n",
        "#   • Bad-posture > 60s alert + “move your body” every 30 minutes\n",
        "#   • Analytics dashboard (posture % and timeline)\n",
        "#\n",
        "# Works on desktop and mobile (browser webcam). For real-time streaming we use streamlit-webrtc."
      ],
      "metadata": {
        "id": "7I7dnjvuaJKp"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "execution_count": 1,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "sXEbENvHSehz",
        "outputId": "2c63bc57-1c1b-44b4-ffdc-447782f4b071"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting mediapipe\n",
            "  Downloading mediapipe-0.10.21-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (9.7 kB)\n",
            "Requirement already satisfied: absl-py in /usr/local/lib/python3.12/dist-packages (from mediapipe) (1.4.0)\n",
            "Requirement already satisfied: attrs>=19.1.0 in /usr/local/lib/python3.12/dist-packages (from mediapipe) (25.4.0)\n",
            "Requirement already satisfied: flatbuffers>=2.0 in /usr/local/lib/python3.12/dist-packages (from mediapipe) (25.9.23)\n",
            "Requirement already satisfied: jax in /usr/local/lib/python3.12/dist-packages (from mediapipe) (0.7.2)\n",
            "Requirement already satisfied: jaxlib in /usr/local/lib/python3.12/dist-packages (from mediapipe) (0.7.2)\n",
            "Requirement already satisfied: matplotlib in /usr/local/lib/python3.12/dist-packages (from mediapipe) (3.10.0)\n",
            "Collecting numpy<2 (from mediapipe)\n",
            "  Downloading numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m61.0/61.0 kB\u001b[0m \u001b[31m1.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: opencv-contrib-python in /usr/local/lib/python3.12/dist-packages (from mediapipe) (4.12.0.88)\n",
            "Collecting protobuf<5,>=4.25.3 (from mediapipe)\n",
            "  Downloading protobuf-4.25.8-cp37-abi3-manylinux2014_x86_64.whl.metadata (541 bytes)\n",
            "Collecting sounddevice>=0.4.4 (from mediapipe)\n",
            "  Downloading sounddevice-0.5.3-py3-none-any.whl.metadata (1.6 kB)\n",
            "Requirement already satisfied: sentencepiece in /usr/local/lib/python3.12/dist-packages (from mediapipe) (0.2.1)\n",
            "Requirement already satisfied: CFFI>=1.0 in /usr/local/lib/python3.12/dist-packages (from sounddevice>=0.4.4->mediapipe) (2.0.0)\n",
            "Requirement already satisfied: ml_dtypes>=0.5.0 in /usr/local/lib/python3.12/dist-packages (from jax->mediapipe) (0.5.3)\n",
            "INFO: pip is looking at multiple versions of jax to determine which version is compatible with other requirements. This could take a while.\n",
            "Collecting jax (from mediapipe)\n",
            "  Downloading jax-0.8.0-py3-none-any.whl.metadata (13 kB)\n",
            "Collecting jaxlib (from mediapipe)\n",
            "  Downloading jaxlib-0.8.0-cp312-cp312-manylinux_2_27_x86_64.whl.metadata (1.3 kB)\n",
            "Collecting jax (from mediapipe)\n",
            "  Downloading jax-0.7.1-py3-none-any.whl.metadata (13 kB)\n",
            "Collecting jaxlib (from mediapipe)\n",
            "  Downloading jaxlib-0.7.1-cp312-cp312-manylinux_2_27_x86_64.whl.metadata (1.3 kB)\n",
            "Requirement already satisfied: opt_einsum in /usr/local/lib/python3.12/dist-packages (from jax->mediapipe) (3.4.0)\n",
            "Requirement already satisfied: scipy>=1.12 in /usr/local/lib/python3.12/dist-packages (from jax->mediapipe) (1.16.3)\n",
            "Requirement already satisfied: contourpy>=1.0.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mediapipe) (1.3.3)\n",
            "Requirement already satisfied: cycler>=0.10 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mediapipe) (0.12.1)\n",
            "Requirement already satisfied: fonttools>=4.22.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mediapipe) (4.60.1)\n",
            "Requirement already satisfied: kiwisolver>=1.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mediapipe) (1.4.9)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mediapipe) (25.0)\n",
            "Requirement already satisfied: pillow>=8 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mediapipe) (11.3.0)\n",
            "Requirement already satisfied: pyparsing>=2.3.1 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mediapipe) (3.2.5)\n",
            "Requirement already satisfied: python-dateutil>=2.7 in /usr/local/lib/python3.12/dist-packages (from matplotlib->mediapipe) (2.9.0.post0)\n",
            "INFO: pip is looking at multiple versions of opencv-contrib-python to determine which version is compatible with other requirements. This could take a while.\n",
            "Collecting opencv-contrib-python (from mediapipe)\n",
            "  Downloading opencv_contrib_python-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (20 kB)\n",
            "Requirement already satisfied: pycparser in /usr/local/lib/python3.12/dist-packages (from CFFI>=1.0->sounddevice>=0.4.4->mediapipe) (2.23)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.7->matplotlib->mediapipe) (1.17.0)\n",
            "Downloading mediapipe-0.10.21-cp312-cp312-manylinux_2_28_x86_64.whl (35.6 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m35.6/35.6 MB\u001b[0m \u001b[31m42.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading numpy-1.26.4-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.0 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m18.0/18.0 MB\u001b[0m \u001b[31m28.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading protobuf-4.25.8-cp37-abi3-manylinux2014_x86_64.whl (294 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m294.9/294.9 kB\u001b[0m \u001b[31m20.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading sounddevice-0.5.3-py3-none-any.whl (32 kB)\n",
            "Downloading jax-0.7.1-py3-none-any.whl (2.8 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m2.8/2.8 MB\u001b[0m \u001b[31m80.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading jaxlib-0.7.1-cp312-cp312-manylinux_2_27_x86_64.whl (81.2 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m81.2/81.2 MB\u001b[0m \u001b[31m8.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading opencv_contrib_python-4.11.0.86-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (69.1 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m69.1/69.1 MB\u001b[0m \u001b[31m10.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: protobuf, numpy, sounddevice, opencv-contrib-python, jaxlib, jax, mediapipe\n",
            "  Attempting uninstall: protobuf\n",
            "    Found existing installation: protobuf 5.29.5\n",
            "    Uninstalling protobuf-5.29.5:\n",
            "      Successfully uninstalled protobuf-5.29.5\n",
            "  Attempting uninstall: numpy\n",
            "    Found existing installation: numpy 2.0.2\n",
            "    Uninstalling numpy-2.0.2:\n",
            "      Successfully uninstalled numpy-2.0.2\n",
            "  Attempting uninstall: opencv-contrib-python\n",
            "    Found existing installation: opencv-contrib-python 4.12.0.88\n",
            "    Uninstalling opencv-contrib-python-4.12.0.88:\n",
            "      Successfully uninstalled opencv-contrib-python-4.12.0.88\n",
            "  Attempting uninstall: jaxlib\n",
            "    Found existing installation: jaxlib 0.7.2\n",
            "    Uninstalling jaxlib-0.7.2:\n",
            "      Successfully uninstalled jaxlib-0.7.2\n",
            "  Attempting uninstall: jax\n",
            "    Found existing installation: jax 0.7.2\n",
            "    Uninstalling jax-0.7.2:\n",
            "      Successfully uninstalled jax-0.7.2\n",
            "\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
            "opencv-python-headless 4.12.0.88 requires numpy<2.3.0,>=2; python_version >= \"3.9\", but you have numpy 1.26.4 which is incompatible.\n",
            "grpcio-status 1.71.2 requires protobuf<6.0dev,>=5.26.1, but you have protobuf 4.25.8 which is incompatible.\n",
            "pytensor 2.35.1 requires numpy>=2.0, but you have numpy 1.26.4 which is incompatible.\n",
            "opencv-python 4.12.0.88 requires numpy<2.3.0,>=2; python_version >= \"3.9\", but you have numpy 1.26.4 which is incompatible.\n",
            "opentelemetry-proto 1.37.0 requires protobuf<7.0,>=5.0, but you have protobuf 4.25.8 which is incompatible.\n",
            "thinc 8.3.6 requires numpy<3.0.0,>=2.0.0, but you have numpy 1.26.4 which is incompatible.\n",
            "ydf 0.13.0 requires protobuf<7.0.0,>=5.29.1, but you have protobuf 4.25.8 which is incompatible.\u001b[0m\u001b[31m\n",
            "\u001b[0mSuccessfully installed jax-0.7.1 jaxlib-0.7.1 mediapipe-0.10.21 numpy-1.26.4 opencv-contrib-python-4.11.0.86 protobuf-4.25.8 sounddevice-0.5.3\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "application/vnd.colab-display-data+json": {
              "pip_warning": {
                "packages": [
                  "google",
                  "numpy"
                ]
              },
              "id": "1173fa3ff9184a468085b5453f71c1b4"
            }
          },
          "metadata": {}
        }
      ],
      "source": [
        "pip install mediapipe"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import math\n",
        "import time\n",
        "import collections\n",
        "from dataclasses import dataclass, field\n",
        "from typing import Deque, Dict, List, Optional, Tuple"
      ],
      "metadata": {
        "id": "EoVCiIl7SiVR"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "pip install av"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "qA4EwtKRabEh",
        "outputId": "545043a0-7cf4-4e1e-96f3-8bf53028d104"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting av\n",
            "  Downloading av-16.0.1-cp312-cp312-manylinux_2_28_x86_64.whl.metadata (4.6 kB)\n",
            "Downloading av-16.0.1-cp312-cp312-manylinux_2_28_x86_64.whl (40.5 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m40.5/40.5 MB\u001b[0m \u001b[31m11.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: av\n",
            "Successfully installed av-16.0.1\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install streamlit"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GFRRvoSDai_E",
        "outputId": "b88c5413-1975-4138-bbf7-721d46195f97"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting streamlit\n",
            "  Downloading streamlit-1.51.0-py3-none-any.whl.metadata (9.5 kB)\n",
            "Requirement already satisfied: altair!=5.4.0,!=5.4.1,<6,>=4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (5.5.0)\n",
            "Requirement already satisfied: blinker<2,>=1.5.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (1.9.0)\n",
            "Requirement already satisfied: cachetools<7,>=4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (5.5.2)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (8.3.0)\n",
            "Requirement already satisfied: numpy<3,>=1.23 in /usr/local/lib/python3.12/dist-packages (from streamlit) (1.26.4)\n",
            "Requirement already satisfied: packaging<26,>=20 in /usr/local/lib/python3.12/dist-packages (from streamlit) (25.0)\n",
            "Requirement already satisfied: pandas<3,>=1.4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (2.2.2)\n",
            "Requirement already satisfied: pillow<13,>=7.1.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (11.3.0)\n",
            "Requirement already satisfied: protobuf<7,>=3.20 in /usr/local/lib/python3.12/dist-packages (from streamlit) (4.25.8)\n",
            "Requirement already satisfied: pyarrow<22,>=7.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (18.1.0)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.12/dist-packages (from streamlit) (2.32.4)\n",
            "Requirement already satisfied: tenacity<10,>=8.1.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (8.5.0)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.12/dist-packages (from streamlit) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit) (4.15.0)\n",
            "Requirement already satisfied: watchdog<7,>=2.1.5 in /usr/local/lib/python3.12/dist-packages (from streamlit) (6.0.0)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.12/dist-packages (from streamlit) (3.1.45)\n",
            "Collecting pydeck<1,>=0.8.0b4 (from streamlit)\n",
            "  Downloading pydeck-0.9.1-py2.py3-none-any.whl.metadata (4.1 kB)\n",
            "Requirement already satisfied: tornado!=6.5.0,<7,>=6.0.3 in /usr/local/lib/python3.12/dist-packages (from streamlit) (6.5.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (3.1.6)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.12/dist-packages (from altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (4.25.1)\n",
            "Requirement already satisfied: narwhals>=1.14.2 in /usr/local/lib/python3.12/dist-packages (from altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (2.10.0)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.12/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.12)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas<3,>=1.4.0->streamlit) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas<3,>=1.4.0->streamlit) (2025.2)\n",
            "Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (3.4.4)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (3.11)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (2.5.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit) (2025.10.5)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.12/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.2)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (3.0.3)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (25.4.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (2025.9.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (0.37.0)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit) (0.28.0)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit) (1.17.0)\n",
            "Downloading streamlit-1.51.0-py3-none-any.whl (10.2 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m10.2/10.2 MB\u001b[0m \u001b[31m69.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading pydeck-0.9.1-py2.py3-none-any.whl (6.9 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m6.9/6.9 MB\u001b[0m \u001b[31m88.9 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: pydeck, streamlit\n",
            "Successfully installed pydeck-0.9.1 streamlit-1.51.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install streamlit-webrtc"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 1000
        },
        "id": "sxHQnO7ZKndV",
        "outputId": "bd79cbab-8d04-4d73-99f9-56c1b8442ac2"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting streamlit-webrtc\n",
            "  Downloading streamlit_webrtc-0.63.11-py3-none-any.whl.metadata (18 kB)\n",
            "Collecting aioice>=0.10.1 (from streamlit-webrtc)\n",
            "  Downloading aioice-0.10.1-py3-none-any.whl.metadata (4.1 kB)\n",
            "Collecting aiortc>=1.11.0 (from streamlit-webrtc)\n",
            "  Downloading aiortc-1.14.0-py3-none-any.whl.metadata (4.9 kB)\n",
            "Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.12/dist-packages (from streamlit-webrtc) (25.0)\n",
            "Requirement already satisfied: streamlit>=0.89.0 in /usr/local/lib/python3.12/dist-packages (from streamlit-webrtc) (1.51.0)\n",
            "Collecting dnspython>=2.0.0 (from aioice>=0.10.1->streamlit-webrtc)\n",
            "  Downloading dnspython-2.8.0-py3-none-any.whl.metadata (5.7 kB)\n",
            "Collecting ifaddr>=0.2.0 (from aioice>=0.10.1->streamlit-webrtc)\n",
            "  Downloading ifaddr-0.2.0-py3-none-any.whl.metadata (4.9 kB)\n",
            "Requirement already satisfied: av<17.0.0,>=14.0.0 in /usr/local/lib/python3.12/dist-packages (from aiortc>=1.11.0->streamlit-webrtc) (16.0.1)\n",
            "Collecting cryptography>=44.0.0 (from aiortc>=1.11.0->streamlit-webrtc)\n",
            "  Downloading cryptography-46.0.3-cp311-abi3-manylinux_2_34_x86_64.whl.metadata (5.7 kB)\n",
            "Requirement already satisfied: google-crc32c>=1.1 in /usr/local/lib/python3.12/dist-packages (from aiortc>=1.11.0->streamlit-webrtc) (1.7.1)\n",
            "Collecting pyee>=13.0.0 (from aiortc>=1.11.0->streamlit-webrtc)\n",
            "  Downloading pyee-13.0.0-py3-none-any.whl.metadata (2.9 kB)\n",
            "Collecting pylibsrtp>=0.10.0 (from aiortc>=1.11.0->streamlit-webrtc)\n",
            "  Downloading pylibsrtp-1.0.0-cp310-abi3-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl.metadata (4.0 kB)\n",
            "Collecting pyopenssl>=25.0.0 (from aiortc>=1.11.0->streamlit-webrtc)\n",
            "  Downloading pyopenssl-25.3.0-py3-none-any.whl.metadata (17 kB)\n",
            "Requirement already satisfied: altair!=5.4.0,!=5.4.1,<6,>=4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit>=0.89.0->streamlit-webrtc) (5.5.0)\n",
            "Requirement already satisfied: blinker<2,>=1.5.0 in /usr/local/lib/python3.12/dist-packages (from streamlit>=0.89.0->streamlit-webrtc) (1.9.0)\n",
            "Requirement already satisfied: cachetools<7,>=4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit>=0.89.0->streamlit-webrtc) (5.5.2)\n",
            "Requirement already satisfied: click<9,>=7.0 in /usr/local/lib/python3.12/dist-packages (from streamlit>=0.89.0->streamlit-webrtc) (8.3.0)\n",
            "Requirement already satisfied: numpy<3,>=1.23 in /usr/local/lib/python3.12/dist-packages (from streamlit>=0.89.0->streamlit-webrtc) (1.26.4)\n",
            "Requirement already satisfied: pandas<3,>=1.4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit>=0.89.0->streamlit-webrtc) (2.2.2)\n",
            "Requirement already satisfied: pillow<13,>=7.1.0 in /usr/local/lib/python3.12/dist-packages (from streamlit>=0.89.0->streamlit-webrtc) (11.3.0)\n",
            "Requirement already satisfied: protobuf<7,>=3.20 in /usr/local/lib/python3.12/dist-packages (from streamlit>=0.89.0->streamlit-webrtc) (4.25.8)\n",
            "Requirement already satisfied: pyarrow<22,>=7.0 in /usr/local/lib/python3.12/dist-packages (from streamlit>=0.89.0->streamlit-webrtc) (18.1.0)\n",
            "Requirement already satisfied: requests<3,>=2.27 in /usr/local/lib/python3.12/dist-packages (from streamlit>=0.89.0->streamlit-webrtc) (2.32.4)\n",
            "Requirement already satisfied: tenacity<10,>=8.1.0 in /usr/local/lib/python3.12/dist-packages (from streamlit>=0.89.0->streamlit-webrtc) (8.5.0)\n",
            "Requirement already satisfied: toml<2,>=0.10.1 in /usr/local/lib/python3.12/dist-packages (from streamlit>=0.89.0->streamlit-webrtc) (0.10.2)\n",
            "Requirement already satisfied: typing-extensions<5,>=4.4.0 in /usr/local/lib/python3.12/dist-packages (from streamlit>=0.89.0->streamlit-webrtc) (4.15.0)\n",
            "Requirement already satisfied: watchdog<7,>=2.1.5 in /usr/local/lib/python3.12/dist-packages (from streamlit>=0.89.0->streamlit-webrtc) (6.0.0)\n",
            "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in /usr/local/lib/python3.12/dist-packages (from streamlit>=0.89.0->streamlit-webrtc) (3.1.45)\n",
            "Requirement already satisfied: pydeck<1,>=0.8.0b4 in /usr/local/lib/python3.12/dist-packages (from streamlit>=0.89.0->streamlit-webrtc) (0.9.1)\n",
            "Requirement already satisfied: tornado!=6.5.0,<7,>=6.0.3 in /usr/local/lib/python3.12/dist-packages (from streamlit>=0.89.0->streamlit-webrtc) (6.5.1)\n",
            "Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit>=0.89.0->streamlit-webrtc) (3.1.6)\n",
            "Requirement already satisfied: jsonschema>=3.0 in /usr/local/lib/python3.12/dist-packages (from altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit>=0.89.0->streamlit-webrtc) (4.25.1)\n",
            "Requirement already satisfied: narwhals>=1.14.2 in /usr/local/lib/python3.12/dist-packages (from altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit>=0.89.0->streamlit-webrtc) (2.10.0)\n",
            "Requirement already satisfied: cffi>=2.0.0 in /usr/local/lib/python3.12/dist-packages (from cryptography>=44.0.0->aiortc>=1.11.0->streamlit-webrtc) (2.0.0)\n",
            "Requirement already satisfied: gitdb<5,>=4.0.1 in /usr/local/lib/python3.12/dist-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit>=0.89.0->streamlit-webrtc) (4.0.12)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.12/dist-packages (from pandas<3,>=1.4.0->streamlit>=0.89.0->streamlit-webrtc) (2.9.0.post0)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.12/dist-packages (from pandas<3,>=1.4.0->streamlit>=0.89.0->streamlit-webrtc) (2025.2)\n",
            "Requirement already satisfied: tzdata>=2022.7 in /usr/local/lib/python3.12/dist-packages (from pandas<3,>=1.4.0->streamlit>=0.89.0->streamlit-webrtc) (2025.2)\n",
            "Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit>=0.89.0->streamlit-webrtc) (3.4.4)\n",
            "Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit>=0.89.0->streamlit-webrtc) (3.11)\n",
            "Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit>=0.89.0->streamlit-webrtc) (2.5.0)\n",
            "Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests<3,>=2.27->streamlit>=0.89.0->streamlit-webrtc) (2025.10.5)\n",
            "Requirement already satisfied: pycparser in /usr/local/lib/python3.12/dist-packages (from cffi>=2.0.0->cryptography>=44.0.0->aiortc>=1.11.0->streamlit-webrtc) (2.23)\n",
            "Requirement already satisfied: smmap<6,>=3.0.1 in /usr/local/lib/python3.12/dist-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit>=0.89.0->streamlit-webrtc) (5.0.2)\n",
            "Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit>=0.89.0->streamlit-webrtc) (3.0.3)\n",
            "Requirement already satisfied: attrs>=22.2.0 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit>=0.89.0->streamlit-webrtc) (25.4.0)\n",
            "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit>=0.89.0->streamlit-webrtc) (2025.9.1)\n",
            "Requirement already satisfied: referencing>=0.28.4 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit>=0.89.0->streamlit-webrtc) (0.37.0)\n",
            "Requirement already satisfied: rpds-py>=0.7.1 in /usr/local/lib/python3.12/dist-packages (from jsonschema>=3.0->altair!=5.4.0,!=5.4.1,<6,>=4.0->streamlit>=0.89.0->streamlit-webrtc) (0.28.0)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.12/dist-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit>=0.89.0->streamlit-webrtc) (1.17.0)\n",
            "Downloading streamlit_webrtc-0.63.11-py3-none-any.whl (220 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m220.2/220.2 kB\u001b[0m \u001b[31m5.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading aioice-0.10.1-py3-none-any.whl (24 kB)\n",
            "Downloading aiortc-1.14.0-py3-none-any.whl (93 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m93.2/93.2 kB\u001b[0m \u001b[31m7.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading cryptography-46.0.3-cp311-abi3-manylinux_2_34_x86_64.whl (4.5 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m4.5/4.5 MB\u001b[0m \u001b[31m37.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading dnspython-2.8.0-py3-none-any.whl (331 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m331.1/331.1 kB\u001b[0m \u001b[31m3.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading ifaddr-0.2.0-py3-none-any.whl (12 kB)\n",
            "Downloading pyee-13.0.0-py3-none-any.whl (15 kB)\n",
            "Downloading pylibsrtp-1.0.0-cp310-abi3-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl (2.4 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m2.4/2.4 MB\u001b[0m \u001b[31m76.6 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading pyopenssl-25.3.0-py3-none-any.whl (57 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m57.3/57.3 kB\u001b[0m \u001b[31m4.2 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: ifaddr, pyee, dnspython, pylibsrtp, cryptography, aioice, pyopenssl, aiortc, streamlit-webrtc\n",
            "  Attempting uninstall: cryptography\n",
            "    Found existing installation: cryptography 43.0.3\n",
            "    Uninstalling cryptography-43.0.3:\n",
            "      Successfully uninstalled cryptography-43.0.3\n",
            "  Attempting uninstall: pyopenssl\n",
            "    Found existing installation: pyOpenSSL 24.2.1\n",
            "    Uninstalling pyOpenSSL-24.2.1:\n",
            "      Successfully uninstalled pyOpenSSL-24.2.1\n",
            "\u001b[31mERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.\n",
            "pydrive2 1.21.3 requires cryptography<44, but you have cryptography 46.0.3 which is incompatible.\n",
            "pydrive2 1.21.3 requires pyOpenSSL<=24.2.1,>=19.1.0, but you have pyopenssl 25.3.0 which is incompatible.\u001b[0m\u001b[31m\n",
            "\u001b[0mSuccessfully installed aioice-0.10.1 aiortc-1.14.0 cryptography-46.0.3 dnspython-2.8.0 ifaddr-0.2.0 pyee-13.0.0 pylibsrtp-1.0.0 pyopenssl-25.3.0 streamlit-webrtc-0.63.11\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "application/vnd.colab-display-data+json": {
              "pip_warning": {
                "packages": [
                  "_openssl",
                  "cryptography"
                ]
              },
              "id": "862b4b0b61d3423dbecbced03cbd179a"
            }
          },
          "metadata": {}
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import av\n",
        "import cv2\n",
        "import numpy as np\n",
        "import streamlit as st\n",
        "from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase\n",
        "\n",
        "import mediapipe as mp\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Rm9g-OgYTD8O",
        "outputId": "03755a0f-51a1-4859-d882-577aeadc4158"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2025-11-02 06:05:02.921 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager\n",
            "2025-11-02 06:05:02.927 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager\n",
            "2025-11-02 06:05:02.935 WARNING streamlit.runtime.caching.cache_data_api: No runtime found, using MemoryCacheStorageManager\n",
            "2025-11-02 06:05:03.312 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:05:03.319 No runtime found, using MemoryCacheStorageManager\n",
            "/usr/local/lib/python3.12/dist-packages/jaxlib/plugin_support.py:71: RuntimeWarning: JAX plugin jax_cuda12_plugin version 0.7.2 is installed, but it is not compatible with the installed jaxlib version 0.7.1, so it will not be used.\n",
            "  warnings.warn(\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "pip install PyQt6"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "OYdK8kh8UlcR",
        "outputId": "50d21549-73db-443e-f076-e154ce13386d"
      },
      "execution_count": 2,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting PyQt6\n",
            "  Downloading pyqt6-6.10.0-1-cp39-abi3-manylinux_2_34_x86_64.whl.metadata (2.1 kB)\n",
            "Collecting PyQt6-sip<14,>=13.8 (from PyQt6)\n",
            "  Downloading pyqt6_sip-13.10.2-cp312-cp312-manylinux_2_5_x86_64.manylinux1_x86_64.whl.metadata (494 bytes)\n",
            "Collecting PyQt6-Qt6<6.11.0,>=6.10.0 (from PyQt6)\n",
            "  Downloading pyqt6_qt6-6.10.0-py3-none-manylinux_2_34_x86_64.whl.metadata (535 bytes)\n",
            "Downloading pyqt6-6.10.0-1-cp39-abi3-manylinux_2_34_x86_64.whl (37.7 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m37.7/37.7 MB\u001b[0m \u001b[31m12.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading pyqt6_qt6-6.10.0-py3-none-manylinux_2_34_x86_64.whl (83.8 MB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m83.8/83.8 MB\u001b[0m \u001b[31m7.1 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hDownloading pyqt6_sip-13.10.2-cp312-cp312-manylinux_2_5_x86_64.manylinux1_x86_64.whl (304 kB)\n",
            "\u001b[2K   \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m304.3/304.3 kB\u001b[0m \u001b[31m21.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hInstalling collected packages: PyQt6-Qt6, PyQt6-sip, PyQt6\n",
            "Successfully installed PyQt6-6.10.0 PyQt6-Qt6-6.10.0 PyQt6-sip-13.10.2\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "from PyQt6 import QtCore, QtGui, QtWidgets"
      ],
      "metadata": {
        "id": "NO1sWgjUTKBO"
      },
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "st.set_page_config(page_title=\"Typing Posture Alert\", page_icon=\"🧍\", layout=\"wide\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sGDmWIotUQ34",
        "outputId": "b5eddbbd-188f-4ce6-dd76-23968b397620"
      },
      "execution_count": 4,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2025-11-02 06:06:01.421 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# ---- GLOBAL CONSTANTS ----"
      ],
      "metadata": {
        "id": "KjH2ZH4LVUtZ"
      },
      "execution_count": 5,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Default thresholds (used before calibration)\n",
        "NECK_TILT_MAX_DEFAULT = 25.0     # deg, larger => slouch\n",
        "BACK_ANGLE_MIN_DEFAULT = 150.0   # deg, smaller => slouch\n",
        "HEAD_PITCH_MAX_DEFAULT = 20.0    # deg, larger => slouch\n",
        "\n",
        "SMOOTH_WINDOW = 30               # frames to smooth decisions (rolling window)\n",
        "BAD_POSTURE_PERSIST_SEC = 60.0   # if bad posture persists > 1 minute => strong alert\n",
        "MOVE_BODY_INTERVAL_SEC = 30 * 60 # every 30 minutes, remind to move\n",
        "\n",
        "# UI scaling for mobile (keep it simple/large)\n",
        "MOBILE_MAX_WIDTH = 720\n"
      ],
      "metadata": {
        "id": "8dX4ugD_WGn2"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# ---- SESSION STATE -------"
      ],
      "metadata": {
        "id": "Cqg5ojTbWZIW"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def ss_get(key, default): #session state\n",
        "    if key not in st.session_state: #check if that vars already exists\n",
        "        st.session_state[key] = default #if key missing creates it again\n",
        "    return st.session_state[key] #returns the value"
      ],
      "metadata": {
        "id": "lv_6YVBWWgXQ"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#save values (val) into steamlit session memory (st.session_state) using key\n",
        "def ss_set(key, val):\n",
        "    st.session_state[key] = val"
      ],
      "metadata": {
        "id": "k56IIM5qWjGn"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Initialize session state\n",
        "ss_get(\"consented\", False) #need users consent\n",
        "ss_get(\"calibrated\", False) #avoid postur detection run before baseline posture is set\n",
        "ss_get(\"baseline_angles\", {\"neck\": None, \"back\": None, \"head\": None}) #empty until the calibration is complete\n",
        "ss_get(\"thresholds\", {\n",
        "    \"NECK_TILT_MAX\": NECK_TILT_MAX_DEFAULT,\n",
        "    \"BACK_ANGLE_MIN\": BACK_ANGLE_MIN_DEFAULT,\n",
        "    \"HEAD_PITCH_MAX\": HEAD_PITCH_MAX_DEFAULT\n",
        "}) # how far u can move before being flag\n",
        "ss_get(\"stats\", {\n",
        "    \"start_ts\": None,           # session start time\n",
        "    \"last_good_ts\": None,       # last time posture was good\n",
        "    \"last_move_reminder\": None, # last time we reminded to move\n",
        "    \"bad_streak_start\": None,   # when a continuous bad posture streak began\n",
        "    \"timeline\": [],             # list of (t_rel_seconds, is_good)\n",
        "    \"good_frames\": 0,\n",
        "    \"bad_frames\": 0,\n",
        "}) #this is a dictionary use to track al posture stats\n",
        "ss_get(\"show_landmarks\", True)\n",
        "ss_get(\"mirror_video\", True)\n",
        "ss_get(\"sensitivity\", 1.0)  # scale thresholds post-calibration, move up if need more sensitivity\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "CxepqqrtWooI",
        "outputId": "951ae389-a013-47c9-e919-c3ad5e2c4a99"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2025-11-02 06:06:10.771 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.772 Session state does not function when running a script without `streamlit run`\n",
            "2025-11-02 06:06:10.773 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.773 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.775 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.776 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.778 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.780 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.781 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.783 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.786 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.790 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.793 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.803 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.807 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.808 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.809 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.810 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.811 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.811 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.812 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.813 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.814 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.815 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.815 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.816 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.817 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.818 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.819 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.819 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.820 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.821 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:06:10.822 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "1.0"
            ]
          },
          "metadata": {},
          "execution_count": 10
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# ---- POSE / GEOMETRY ----"
      ],
      "metadata": {
        "id": "Vs77cbslWrRK"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def _unit(v: np.ndarray) -> np.ndarray: #helper funs _unit\n",
        "  n = np.linalg.norm(v)\n",
        "  return v if n == 0 else v / n #return numpy array"
      ],
      "metadata": {
        "id": "U1kSWYkwXyEz"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:\n",
        "    v1u, v2u = _unit(v1), _unit(v2)\n",
        "    dot = float(np.clip(np.dot(v1u, v2u), -1.0, 1.0))\n",
        "    return float(np.degrees(np.arccos(dot)))"
      ],
      "metadata": {
        "id": "bXt6NwulaHG2"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from typing import Deque, Dict, List, Optional, Tuple"
      ],
      "metadata": {
        "id": "F4FgJV-Nbynp"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#this is the shorter version than before\n",
        "VERTICAL = np.array([0.0, -1.0])\n",
        "\n",
        "def vec(pA, pB, W, H) -> np.ndarray:\n",
        "    return np.array([(pB.x - pA.x) * W, (pB.y - pA.y) * H], dtype=float)\n",
        "\n",
        "def features_from_landmarks(lm, W: int, H: int) -> Tuple[float, float, float]:\n",
        "    \"\"\"\n",
        "    Returns (neck_tilt, back_angle, head_pitch) in degrees.\n",
        "    Uses MediaPipe Pose indices: 11,12=shoulders; 23,24=hips; 0=nose\n",
        "    \"\"\"\n",
        "    ls, rs, lh, rh, nose = lm[11], lm[12], lm[23], lm[24], lm[0]\n",
        "    make_point = lambda x, y: type('P', (), dict(x=x, y=y))()\n",
        "\n",
        "    ear_like = make_point((ls.x + nose.x) / 2, (ls.y + nose.y) / 2)\n",
        "    neck_tilt  = angle_deg(vec(ls, ear_like, W, H), VERTICAL)\n",
        "    back_angle = angle_deg(vec(lh, ls, W, H), VERTICAL)\n",
        "    mid = make_point((ls.x + rs.x) / 2, (ls.y + rs.y) / 2)\n",
        "    head_pitch = angle_deg(vec(mid, nose, W, H), VERTICAL)\n",
        "\n",
        "    return neck_tilt, back_angle, head_pitch\n"
      ],
      "metadata": {
        "id": "1oavJ3ytbXJ7"
      },
      "execution_count": 16,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# ---- VIDEO PROCESSOR ----"
      ],
      "metadata": {
        "id": "5VpztRlNbisA"
      },
      "execution_count": 17,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "from dataclasses import dataclass, field"
      ],
      "metadata": {
        "id": "zFzW7HuncL9F"
      },
      "execution_count": 18,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "@dataclass\n",
        "class PostureState:\n",
        "    hist: Deque[int] = field(default_factory=lambda: collections.deque(maxlen=SMOOTH_WINDOW))\n",
        "    last_frame_ts: float = 0.0\n",
        "\n",
        "class VideoProcessor(VideoProcessorBase):\n",
        "    def __init__(self):\n",
        "        self.pose = mp.solutions.pose.Pose(\n",
        "            min_detection_confidence=0.5, min_tracking_confidence=0.5\n",
        "        )\n",
        "        self.draw = mp.solutions.drawing_utils\n",
        "        self.styles = mp.solutions.drawing_styles\n",
        "        self.state = PostureState()\n",
        "\n",
        "    def _get_thresholds(self) -> Dict[str, float]:\n",
        "        th = st.session_state[\"thresholds\"].copy()\n",
        "        # Apply sensitivity scaling (e.g., 0.8 = stricter, 1.2 = looser)\n",
        "        s = float(st.session_state.get(\"sensitivity\", 1.0))\n",
        "        th[\"NECK_TILT_MAX\"] = th[\"NECK_TILT_MAX\"] * (1.0 / s)   # stricter if s>1\n",
        "        th[\"HEAD_PITCH_MAX\"] = th[\"HEAD_PITCH_MAX\"] * (1.0 / s)\n",
        "        th[\"BACK_ANGLE_MIN\"] = th[\"BACK_ANGLE_MIN\"] * (1.0 / s)\n",
        "        return th\n",
        "\n",
        "    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:\n",
        "        img = frame.to_ndarray(format=\"bgr24\")\n",
        "        H, W = img.shape[:2]\n",
        "\n",
        "        # Mirror for a more natural selfie view if enabled\n",
        "        if st.session_state.get(\"mirror_video\", True):\n",
        "            img = cv2.flip(img, 1)\n",
        "\n",
        "        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
        "        res = self.pose.process(rgb)\n",
        "\n",
        "        neck = back = head = None\n",
        "        is_slouch = False\n",
        "        if res.pose_landmarks:\n",
        "            lm = res.pose_landmarks.landmark\n",
        "            neck, back, head = features_from_landmarks(lm, W, H)\n",
        "\n",
        "            # Draw landmarks if user wants\n",
        "            if st.session_state.get(\"show_landmarks\", True):\n",
        "                self.draw.draw_landmarks(\n",
        "                    img, res.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,\n",
        "                    landmark_drawing_spec=self.styles.get_default_pose_landmarks_style()\n",
        "                )\n",
        "\n",
        "            # Compare to thresholds\n",
        "            th = self._get_thresholds()\n",
        "            bad_neck = (neck is not None and neck > th[\"NECK_TILT_MAX\"])\n",
        "            bad_back = (back is not None and back < th[\"BACK_ANGLE_MIN\"])\n",
        "            bad_head = (head is not None and head > th[\"HEAD_PITCH_MAX\"])\n",
        "\n",
        "            is_slouch = bad_neck or bad_back or bad_head\n",
        "\n",
        "            # Coaching message (pick strongest offender)\n",
        "            msg = \"\"\n",
        "            if is_slouch:\n",
        "                offenders = []\n",
        "                if bad_back: offenders.append((\"back\", abs(back - th[\"BACK_ANGLE_MIN\"])))\n",
        "                if bad_head: offenders.append((\"head\", abs(head - th[\"HEAD_PITCH_MAX\"])))\n",
        "                if bad_neck: offenders.append((\"neck\", abs(neck - th[\"NECK_TILT_MAX\"])))\n",
        "                offenders.sort(key=lambda x: x[1], reverse=True)\n",
        "                top = offenders[0][0] if offenders else None\n",
        "                if top == \"back\":\n",
        "                    msg = \"Adjust your back (sit upright).\"\n",
        "                elif top == \"head\":\n",
        "                    msg = \"Bring your head back (reduce forward head).\"\n",
        "                elif top == \"neck\":\n",
        "                    msg = \"Relax neck; stack ears over shoulders.\"\n",
        "                else:\n",
        "                    msg = \"Adjust posture (sit tall).\"\n",
        "            else:\n",
        "                msg = \"Good posture 👍\"\n",
        "\n",
        "            # Overlay numbers + message\n",
        "            y = 28\n",
        "            def put(txt, color=(0,255,0)):\n",
        "                nonlocal y\n",
        "                cv2.putText(img, txt, (10,y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)\n",
        "                y += 28\n",
        "\n",
        "            if neck is not None:\n",
        "                put(f\"Neck tilt : {neck:5.1f}°\")\n",
        "                put(f\"Back angle: {back:5.1f}°\")\n",
        "                put(f\"Head pitch: {head:5.1f}°\")\n",
        "\n",
        "            if is_slouch:\n",
        "                cv2.rectangle(img, (0,0), (W,60), (0,0,255), -1)\n",
        "                cv2.putText(img, msg + \"  (1-min timer running)\", (10,42),\n",
        "                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2, cv2.LINE_AA)\n",
        "            else:\n",
        "                cv2.rectangle(img, (0,0), (W,40), (0,128,0), -1)\n",
        "                cv2.putText(img, msg, (10,28),\n",
        "                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2, cv2.LINE_AA)\n",
        "\n",
        "        # ---- Update stats / analytics ----\n",
        "        now = time.time()\n",
        "        stats = st.session_state[\"stats\"]\n",
        "        if stats[\"start_ts\"] is None:\n",
        "            stats[\"start_ts\"] = now\n",
        "            stats[\"last_move_reminder\"] = now\n",
        "\n",
        "        # Smoothing\n",
        "        self.state.hist.append(1 if is_slouch else 0)\n",
        "        slouch_ratio = sum(self.state.hist) / max(1, len(self.state.hist))\n",
        "        is_good = (slouch_ratio < 0.5)\n",
        "\n",
        "        # Frame counters (approximate, FPS-independent aggregation)\n",
        "        if is_good:\n",
        "            stats[\"good_frames\"] += 1\n",
        "            stats[\"last_good_ts\"] = now\n",
        "        else:\n",
        "            stats[\"bad_frames\"] += 1\n",
        "\n",
        "        # Continuous bad posture streak timing\n",
        "        if not is_good:\n",
        "            if stats[\"bad_streak_start\"] is None:\n",
        "                stats[\"bad_streak_start\"] = now\n",
        "            else:\n",
        "                if (now - stats[\"bad_streak_start\"]) >= BAD_POSTURE_PERSIST_SEC:\n",
        "                    # Strong alert (Toast)\n",
        "                    st.toast(\"⏰ Bad posture for 1 minute — please reset posture.\", icon=\"⚠️\")\n",
        "                    stats[\"bad_streak_start\"] = now  # reset the one-minute window\n",
        "        else:\n",
        "            stats[\"bad_streak_start\"] = None\n",
        "\n",
        "        # Move-your-body every 30 minutes\n",
        "        if stats[\"last_move_reminder\"] is not None and (now - stats[\"last_move_reminder\"]) >= MOVE_BODY_INTERVAL_SEC:\n",
        "            st.toast(\"🕒 Time to move your body for a minute!\", icon=\"⏳\")\n",
        "            stats[\"last_move_reminder\"] = now\n",
        "\n",
        "        # Timeline (every ~2 seconds to keep it light)\n",
        "        if (now - self.state.last_frame_ts) > 2.0:\n",
        "            t_rel = now - stats[\"start_ts\"]\n",
        "            stats[\"timeline\"].append((t_rel, 1 if is_good else 0))\n",
        "            self.state.last_frame_ts = now\n",
        "\n",
        "        # Ship processed frame out\n",
        "        return av.VideoFrame.from_ndarray(img, format=\"bgr24\")\n",
        "\n"
      ],
      "metadata": {
        "id": "UQzt4P0Dbf0-"
      },
      "execution_count": 20,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# ---- CALIBRATION --------"
      ],
      "metadata": {
        "id": "1TUeTqsKcEkq"
      },
      "execution_count": 21,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def run_calibration(seconds: int = 5):\n",
        "    \"\"\"\n",
        "    Collects a few seconds of 'good posture' frames and sets personalized thresholds.\n",
        "    \"\"\"\n",
        "    st.info(\"Sit in your best posture. Calibration will average your angles and set personalized thresholds.\")\n",
        "    th0 = {\n",
        "        \"NECK_TILT_MAX\": NECK_TILT_MAX_DEFAULT,\n",
        "        \"BACK_ANGLE_MIN\": BACK_ANGLE_MIN_DEFAULT,\n",
        "        \"HEAD_PITCH_MAX\": HEAD_PITCH_MAX_DEFAULT,\n",
        "    }\n",
        "\n",
        "    # Use a tiny temporary processor to read angles for a few seconds\n",
        "    mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)\n",
        "    draw = mp.solutions.drawing_utils\n",
        "\n",
        "    st.write(\"Click **Start camera** below, then hold still in good posture.\")\n",
        "    calib_ctx = webrtc_streamer(\n",
        "        key=\"calibration\",\n",
        "        mode=WebRtcMode.SENDRECV,\n",
        "        media_stream_constraints={\"video\": True, \"audio\": False},\n",
        "        video_processor_factory=None,  # raw frames\n",
        "    )\n",
        "\n",
        "    readings = []\n",
        "    start = None\n",
        "    progress = st.progress(0.0, text=\"Waiting for camera...\")\n",
        "    while calib_ctx.state.playing:\n",
        "        frame = calib_ctx.video_receiver.get_frame(timeout=1)\n",
        "        if frame is None:\n",
        "            continue\n",
        "        img = frame.to_ndarray(format=\"bgr24\")\n",
        "        H, W = img.shape[:2]\n",
        "        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)\n",
        "        res = mp_pose.process(rgb)\n",
        "        if res.pose_landmarks:\n",
        "            lm = res.pose_landmarks.landmark\n",
        "            neck, back, head = features_from_landmarks(lm, W, H)\n",
        "            readings.append((neck, back, head))\n",
        "            if start is None:\n",
        "                start = time.time()\n",
        "            elapsed = time.time() - start\n",
        "            progress.progress(min(1.0, elapsed/seconds), text=f\"Calibrating... {min(seconds, int(elapsed))}/{seconds}s\")\n",
        "            if elapsed >= seconds:\n",
        "                break\n",
        "\n",
        "    # Compute personalized thresholds\n",
        "    if readings:\n",
        "        arr = np.array(readings)  # shape (n, 3)\n",
        "        neck_mean, back_mean, head_mean = arr.mean(axis=0).tolist()\n",
        "        ss_set(\"baseline_angles\", {\"neck\": neck_mean, \"back\": back_mean, \"head\": head_mean})\n",
        "\n",
        "        # Set thresholds relative to baseline with safe margins\n",
        "        personalized = {\n",
        "            \"NECK_TILT_MAX\": max(10.0, neck_mean + 10.0),  # allow +10° from your best\n",
        "            \"BACK_ANGLE_MIN\": min(179.0, back_mean - 10.0), # allow -10° from your best\n",
        "            \"HEAD_PITCH_MAX\": max(8.0, head_mean + 8.0),    # allow +8° from your best\n",
        "        }\n",
        "        ss_set(\"thresholds\", personalized)\n",
        "        ss_set(\"calibrated\", True)\n",
        "        st.success(\"Calibration complete! Personalized thresholds set.\")\n",
        "    else:\n",
        "        st.warning(\"No pose detected during calibration. You can skip calibration or try again.\")\n"
      ],
      "metadata": {
        "id": "OwyFZ9ZBcgCq"
      },
      "execution_count": 22,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# ---- UI LAYOUT ---------"
      ],
      "metadata": {
        "id": "Zdr0H6nCckKv"
      },
      "execution_count": 23,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def consent_gate():\n",
        "    st.title(\"🧍 Typing Posture Alert\")\n",
        "    st.markdown(\n",
        "        \"\"\"\n",
        "        **This app uses your webcam in your browser** to analyze your posture locally (no video is uploaded).\n",
        "        Please confirm:\n",
        "        - I allow the app to access my webcam.\n",
        "        - I understand posture analysis runs in this session only and is not stored.\n",
        "        \"\"\"\n",
        "    )\n",
        "    agree = st.checkbox(\"I consent to webcam access and local posture analysis.\")\n",
        "    if st.button(\"Continue\", type=\"primary\", disabled=not agree):\n",
        "        ss_set(\"consented\", True)\n",
        "        st.rerun()\n",
        "\n",
        "def monitor_section():\n",
        "    st.header(\"Live Monitor\")\n",
        "    colL, colR = st.columns([2,1])\n",
        "    with colL:\n",
        "        st.toggle(\"Mirror video\", value=st.session_state[\"mirror_video\"], key=\"mirror_video\")\n",
        "        st.toggle(\"Show landmarks\", value=st.session_state[\"show_landmarks\"], key=\"show_landmarks\")\n",
        "        st.slider(\"Sensitivity (higher = stricter)\", 0.7, 1.5, key=\"sensitivity\", value=1.0, step=0.05)\n",
        "        webrtc_ctx = webrtc_streamer(\n",
        "            key=\"posture-monitor\",\n",
        "            mode=WebRtcMode.SENDRECV,\n",
        "            media_stream_constraints={\"video\": True, \"audio\": False},\n",
        "            video_processor_factory=VideoProcessor,\n",
        "        )\n",
        "\n",
        "    with colR:\n",
        "        st.subheader(\"Analytics\")\n",
        "        stats = st.session_state[\"stats\"]\n",
        "        start_ts = stats[\"start_ts\"]\n",
        "        good = stats[\"good_frames\"]\n",
        "        bad = stats[\"bad_frames\"]\n",
        "        total = max(1, good + bad)\n",
        "        pct = 100.0 * good / total\n",
        "        st.metric(\"Posture % Good\", f\"{pct:.1f}%\")\n",
        "        st.caption(\"Percentage of frames assessed as 'good' during this session.\")\n",
        "\n",
        "        # Timeline chart\n",
        "        tl = stats[\"timeline\"]\n",
        "        if len(tl) >= 2:\n",
        "            import pandas as pd\n",
        "            df = pd.DataFrame(tl, columns=[\"t_sec\", \"good\"])\n",
        "            df[\"minutes\"] = df[\"t_sec\"] / 60.0\n",
        "            df.set_index(\"minutes\", inplace=True)\n",
        "            st.line_chart(df[\"good\"], height=160, use_container_width=True)\n",
        "            st.caption(\"Timeline: 1 = good posture, 0 = slouching (sampled ~every 2s).\")\n",
        "        else:\n",
        "            st.info(\"Timeline will appear after ~10 seconds of monitoring.\")\n",
        "\n",
        "        # Quick stats\n",
        "        if start_ts:\n",
        "            elapsed_min = (time.time() - start_ts) / 60.0\n",
        "            st.metric(\"Session length\", f\"{elapsed_min:.1f} min\")\n",
        "\n",
        "        st.divider()\n",
        "        if st.button(\"Reset analytics\"):\n",
        "            ss_set(\"stats\", {\n",
        "                \"start_ts\": None,\n",
        "                \"last_good_ts\": None,\n",
        "                \"last_move_reminder\": None,\n",
        "                \"bad_streak_start\": None,\n",
        "                \"timeline\": [],\n",
        "                \"good_frames\": 0,\n",
        "                \"bad_frames\": 0,\n",
        "            })\n",
        "            st.toast(\"Analytics reset.\", icon=\"🔄\")\n",
        "            st.rerun()\n",
        "\n",
        "    st.info(\"Coaching: If you see red banner, adjust your **back / head / neck or distance**. \"\n",
        "            \"If bad posture persists for 1 minute, you’ll get a strong reminder. \"\n",
        "            \"Every 30 minutes, you’ll be nudged to move your body.\")\n",
        "\n",
        "\n",
        "def sidebar_info():\n",
        "    with st.sidebar:\n",
        "        st.header(\"Calibration\")\n",
        "        if not st.session_state[\"calibrated\"]:\n",
        "            st.button(\"Run 5s calibration\", on_click=lambda: None, key=\"calib_enabler\")\n",
        "            if st.session_state.get(\"calib_enabler_clicked_once\") is None:\n",
        "                st.session_state[\"calib_enabler_clicked_once\"] = True\n",
        "            run_button = st.button(\"Start calibration now\")\n",
        "            if run_button:\n",
        "                run_calibration(5)\n",
        "        else:\n",
        "            base = st.session_state[\"baseline_angles\"]\n",
        "            th = st.session_state[\"thresholds\"]\n",
        "            st.success(\"Calibrated ✅\")\n",
        "            st.write(\"**Baseline angles (your best):**\")\n",
        "            st.json({k: round(v,1) if v else None for k,v in base.items()})\n",
        "            st.write(\"**Personalized thresholds:**\")\n",
        "            st.json({k: round(v,1) for k,v in th.items()})\n",
        "            if st.button(\"Re-calibrate\"):\n",
        "                ss_set(\"calibrated\", False)\n",
        "\n",
        "        st.divider()\n",
        "        st.subheader(\"Privacy\")\n",
        "        st.caption(\"• Video stays in your browser session.\\n\"\n",
        "                   \"• No dataset is collected or stored.\\n\"\n",
        "                   \"• You can close the tab to end the session.\")"
      ],
      "metadata": {
        "id": "IKKwQP54cmzJ"
      },
      "execution_count": 24,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# ---- MAIN APP FLOW ------"
      ],
      "metadata": {
        "id": "Vc2ing4icq38"
      },
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "def main():\n",
        "    # Mobile width hint\n",
        "    st.markdown(\n",
        "        f\"\"\"\n",
        "        <style>\n",
        "        .block-container {{ max-width: {MOBILE_MAX_WIDTH}px; }}\n",
        "        </style>\n",
        "        \"\"\",\n",
        "        unsafe_allow_html=True\n",
        "    )\n",
        "\n",
        "    if not st.session_state[\"consented\"]:\n",
        "        consent_gate()\n",
        "        return\n",
        "\n",
        "    # Content\n",
        "    st.title(\"🧍 Typing Posture Alert\")\n",
        "    st.caption(\"Webcam + MediaPipe detects slouching posture and gently alerts. \"\n",
        "               \"Calibrate once, then monitor your study session.\")\n",
        "\n",
        "    sidebar_info()\n",
        "    monitor_section()\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    main()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7XUxZ6QXcsrt",
        "outputId": "a1cd62c0-bc77-4a3c-fd5d-ff2ead158cf5"
      },
      "execution_count": 26,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "2025-11-02 06:12:29.750 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.064 \n",
            "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
            "  command:\n",
            "\n",
            "    streamlit run /usr/local/lib/python3.12/dist-packages/colab_kernel_launcher.py [ARGUMENTS]\n",
            "2025-11-02 06:12:30.066 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.067 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.068 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.069 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.070 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.072 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.073 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.074 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.075 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.076 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.077 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.078 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.079 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.080 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.081 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.082 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.083 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.084 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.085 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.086 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
            "2025-11-02 06:12:30.087 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!jupyter nbconvert --to script ML_project_posture_alert.ipynb\n"
      ],
      "metadata": {
        "id": "53sJQ0wocvKz",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "ad70b6d2-0f38-41c1-813d-52d472753723"
      },
      "execution_count": 27,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "[NbConvertApp] WARNING | pattern 'ML_project_posture_alert.ipynb' matched no files\n",
            "This application is used to convert notebook files (*.ipynb)\n",
            "        to various other formats.\n",
            "\n",
            "        WARNING: THE COMMANDLINE INTERFACE MAY CHANGE IN FUTURE RELEASES.\n",
            "\n",
            "Options\n",
            "=======\n",
            "The options below are convenience aliases to configurable class-options,\n",
            "as listed in the \"Equivalent to\" description-line of the aliases.\n",
            "To see all configurable class-options for some <cmd>, use:\n",
            "    <cmd> --help-all\n",
            "\n",
            "--debug\n",
            "    set log level to logging.DEBUG (maximize logging output)\n",
            "    Equivalent to: [--Application.log_level=10]\n",
            "--show-config\n",
            "    Show the application's configuration (human-readable format)\n",
            "    Equivalent to: [--Application.show_config=True]\n",
            "--show-config-json\n",
            "    Show the application's configuration (json format)\n",
            "    Equivalent to: [--Application.show_config_json=True]\n",
            "--generate-config\n",
            "    generate default config file\n",
            "    Equivalent to: [--JupyterApp.generate_config=True]\n",
            "-y\n",
            "    Answer yes to any questions instead of prompting.\n",
            "    Equivalent to: [--JupyterApp.answer_yes=True]\n",
            "--execute\n",
            "    Execute the notebook prior to export.\n",
            "    Equivalent to: [--ExecutePreprocessor.enabled=True]\n",
            "--allow-errors\n",
            "    Continue notebook execution even if one of the cells throws an error and include the error message in the cell output (the default behaviour is to abort conversion). This flag is only relevant if '--execute' was specified, too.\n",
            "    Equivalent to: [--ExecutePreprocessor.allow_errors=True]\n",
            "--stdin\n",
            "    read a single notebook file from stdin. Write the resulting notebook with default basename 'notebook.*'\n",
            "    Equivalent to: [--NbConvertApp.from_stdin=True]\n",
            "--stdout\n",
            "    Write notebook output to stdout instead of files.\n",
            "    Equivalent to: [--NbConvertApp.writer_class=StdoutWriter]\n",
            "--inplace\n",
            "    Run nbconvert in place, overwriting the existing notebook (only\n",
            "            relevant when converting to notebook format)\n",
            "    Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory=]\n",
            "--clear-output\n",
            "    Clear output of current file and save in place,\n",
            "            overwriting the existing notebook.\n",
            "    Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --ClearOutputPreprocessor.enabled=True]\n",
            "--coalesce-streams\n",
            "    Coalesce consecutive stdout and stderr outputs into one stream (within each cell).\n",
            "    Equivalent to: [--NbConvertApp.use_output_suffix=False --NbConvertApp.export_format=notebook --FilesWriter.build_directory= --CoalesceStreamsPreprocessor.enabled=True]\n",
            "--no-prompt\n",
            "    Exclude input and output prompts from converted document.\n",
            "    Equivalent to: [--TemplateExporter.exclude_input_prompt=True --TemplateExporter.exclude_output_prompt=True]\n",
            "--no-input\n",
            "    Exclude input cells and output prompts from converted document.\n",
            "            This mode is ideal for generating code-free reports.\n",
            "    Equivalent to: [--TemplateExporter.exclude_output_prompt=True --TemplateExporter.exclude_input=True --TemplateExporter.exclude_input_prompt=True]\n",
            "--allow-chromium-download\n",
            "    Whether to allow downloading chromium if no suitable version is found on the system.\n",
            "    Equivalent to: [--WebPDFExporter.allow_chromium_download=True]\n",
            "--disable-chromium-sandbox\n",
            "    Disable chromium security sandbox when converting to PDF..\n",
            "    Equivalent to: [--WebPDFExporter.disable_sandbox=True]\n",
            "--show-input\n",
            "    Shows code input. This flag is only useful for dejavu users.\n",
            "    Equivalent to: [--TemplateExporter.exclude_input=False]\n",
            "--embed-images\n",
            "    Embed the images as base64 dataurls in the output. This flag is only useful for the HTML/WebPDF/Slides exports.\n",
            "    Equivalent to: [--HTMLExporter.embed_images=True]\n",
            "--sanitize-html\n",
            "    Whether the HTML in Markdown cells and cell outputs should be sanitized..\n",
            "    Equivalent to: [--HTMLExporter.sanitize_html=True]\n",
            "--log-level=<Enum>\n",
            "    Set the log level by value or name.\n",
            "    Choices: any of [0, 10, 20, 30, 40, 50, 'DEBUG', 'INFO', 'WARN', 'ERROR', 'CRITICAL']\n",
            "    Default: 30\n",
            "    Equivalent to: [--Application.log_level]\n",
            "--config=<Unicode>\n",
            "    Full path of a config file.\n",
            "    Default: ''\n",
            "    Equivalent to: [--JupyterApp.config_file]\n",
            "--to=<Unicode>\n",
            "    The export format to be used, either one of the built-in formats\n",
            "            ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf']\n",
            "            or a dotted object name that represents the import path for an\n",
            "            ``Exporter`` class\n",
            "    Default: ''\n",
            "    Equivalent to: [--NbConvertApp.export_format]\n",
            "--template=<Unicode>\n",
            "    Name of the template to use\n",
            "    Default: ''\n",
            "    Equivalent to: [--TemplateExporter.template_name]\n",
            "--template-file=<Unicode>\n",
            "    Name of the template file to use\n",
            "    Default: None\n",
            "    Equivalent to: [--TemplateExporter.template_file]\n",
            "--theme=<Unicode>\n",
            "    Template specific theme(e.g. the name of a JupyterLab CSS theme distributed\n",
            "    as prebuilt extension for the lab template)\n",
            "    Default: 'light'\n",
            "    Equivalent to: [--HTMLExporter.theme]\n",
            "--sanitize_html=<Bool>\n",
            "    Whether the HTML in Markdown cells and cell outputs should be sanitized.This\n",
            "    should be set to True by nbviewer or similar tools.\n",
            "    Default: False\n",
            "    Equivalent to: [--HTMLExporter.sanitize_html]\n",
            "--writer=<DottedObjectName>\n",
            "    Writer class used to write the\n",
            "                                        results of the conversion\n",
            "    Default: 'FilesWriter'\n",
            "    Equivalent to: [--NbConvertApp.writer_class]\n",
            "--post=<DottedOrNone>\n",
            "    PostProcessor class used to write the\n",
            "                                        results of the conversion\n",
            "    Default: ''\n",
            "    Equivalent to: [--NbConvertApp.postprocessor_class]\n",
            "--output=<Unicode>\n",
            "    Overwrite base name use for output files.\n",
            "                Supports pattern replacements '{notebook_name}'.\n",
            "    Default: '{notebook_name}'\n",
            "    Equivalent to: [--NbConvertApp.output_base]\n",
            "--output-dir=<Unicode>\n",
            "    Directory to write output(s) to. Defaults\n",
            "                                  to output to the directory of each notebook. To recover\n",
            "                                  previous default behaviour (outputting to the current\n",
            "                                  working directory) use . as the flag value.\n",
            "    Default: ''\n",
            "    Equivalent to: [--FilesWriter.build_directory]\n",
            "--reveal-prefix=<Unicode>\n",
            "    The URL prefix for reveal.js (version 3.x).\n",
            "            This defaults to the reveal CDN, but can be any url pointing to a copy\n",
            "            of reveal.js.\n",
            "            For speaker notes to work, this must be a relative path to a local\n",
            "            copy of reveal.js: e.g., \"reveal.js\".\n",
            "            If a relative path is given, it must be a subdirectory of the\n",
            "            current directory (from which the server is run).\n",
            "            See the usage documentation\n",
            "            (https://nbconvert.readthedocs.io/en/latest/usage.html#reveal-js-html-slideshow)\n",
            "            for more details.\n",
            "    Default: ''\n",
            "    Equivalent to: [--SlidesExporter.reveal_url_prefix]\n",
            "--nbformat=<Enum>\n",
            "    The nbformat version to write.\n",
            "            Use this to downgrade notebooks.\n",
            "    Choices: any of [1, 2, 3, 4]\n",
            "    Default: 4\n",
            "    Equivalent to: [--NotebookExporter.nbformat_version]\n",
            "\n",
            "Examples\n",
            "--------\n",
            "\n",
            "    The simplest way to use nbconvert is\n",
            "\n",
            "            > jupyter nbconvert mynotebook.ipynb --to html\n",
            "\n",
            "            Options include ['asciidoc', 'custom', 'html', 'latex', 'markdown', 'notebook', 'pdf', 'python', 'qtpdf', 'qtpng', 'rst', 'script', 'slides', 'webpdf'].\n",
            "\n",
            "            > jupyter nbconvert --to latex mynotebook.ipynb\n",
            "\n",
            "            Both HTML and LaTeX support multiple output templates. LaTeX includes\n",
            "            'base', 'article' and 'report'.  HTML includes 'basic', 'lab' and\n",
            "            'classic'. You can specify the flavor of the format used.\n",
            "\n",
            "            > jupyter nbconvert --to html --template lab mynotebook.ipynb\n",
            "\n",
            "            You can also pipe the output to stdout, rather than a file\n",
            "\n",
            "            > jupyter nbconvert mynotebook.ipynb --stdout\n",
            "\n",
            "            PDF is generated via latex\n",
            "\n",
            "            > jupyter nbconvert mynotebook.ipynb --to pdf\n",
            "\n",
            "            You can get (and serve) a Reveal.js-powered slideshow\n",
            "\n",
            "            > jupyter nbconvert myslides.ipynb --to slides --post serve\n",
            "\n",
            "            Multiple notebooks can be given at the command line in a couple of\n",
            "            different ways:\n",
            "\n",
            "            > jupyter nbconvert notebook*.ipynb\n",
            "            > jupyter nbconvert notebook1.ipynb notebook2.ipynb\n",
            "\n",
            "            or you can specify the notebooks list in a config file, containing::\n",
            "\n",
            "                c.NbConvertApp.notebooks = [\"my_notebook.ipynb\"]\n",
            "\n",
            "            > jupyter nbconvert --config mycfg.py\n",
            "\n",
            "To see all available configurables, use `--help-all`.\n",
            "\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "le1lbMCHTZTw"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}