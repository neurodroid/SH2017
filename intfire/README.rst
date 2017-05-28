Grid cell model
===============

This archive contains a spiking continuous attractor model of grid cells found
in medial entorhinal cortex (MEC) of rodents. It was used for modeling the
theta-nested gamma activity and grid firing fields in MEC and the results have
been published in [PASTOLL2013]_. The code in this archive reproduces parts of
the figures found in the published article.

For installation and prerequisites necessary to run the model see the
INSTALL.rst file in this directory.

Some changes to replace constant input currents with Poisson inputs driving
AMPA/NMDA synapses, as published in [SH2017]_.

.. [PASTOLL2013] Pastoll, H., Solanka, L., van Rossum, M. C. W., & Nolan, M. F.
    (2013). Feedback inhibition enables theta-nested gamma oscillations and grid
    firing fields. Neuron, 77(1), 141–154. doi:10.1016/j.neuron.2012.11.032

.. [SH2017] Schmidt-Hieber, C., Toleikyte, G., Aitchison, L., Roth, A., Clark,
    B. A., Branco, T., Hausser, M. (2017) Active dendritic integration as a
    mechanism for robust and precise grid cell firing. Nature Neuroscience,
    in press
