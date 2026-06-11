# wetting-and-bouncing-validation

A collection of simulation videos and codes for contact angle studies, together with validation videos of droplet bouncing on partially wetting surfaces.

## Overview

This repository provides experimental and numerical validation materials for droplet wetting and bouncing phenomena, including comparison videos and Python simulation codes.

## Files

### Coalescence-induced bouncing

- `Coalescence_Experimental_video.mp4`
- `Coalescence_Numerical_video.mp4`

These two videos compare physical experiments and numerical simulations of coalescence-induced droplet bouncing. After two droplets coalesce, the released surface energy is converted into mechanical energy, enabling the merged droplet to bounce off the surface.

### Rebound on partially wetting surface

- `Rebound_on_PWsurface_Experimental_video.mp4`
- `Rebound_on_PWsurface_Numerical_video.mp4`

These two videos compare physical experiments and numerical simulations of droplet rebound on a partially wetting (PW) surface composed of locally superhydrophobic and locally superhydrophilic regions. The droplet rebounds with a rebound angle close to \(0^\circ\).

### Codes

- `PythonCode_for_ContactAngle_pi_over_2.py`  
  Code for simulating the static contact angle of a droplet.

- `PythonCode_for_Rebound_on_PWsurface.py`  
  Code for simulating droplet rebound on a partially wetting (PW) surface.

## Requirements

To run the Python codes, please make sure that the `pysph` library is installed in your Python environment.

## Purpose

This repository is intended to provide supporting materials for validation of numerical simulations related to droplet wetting and bouncing behaviors.