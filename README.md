 # Well-Mixed Case

## Base Case

Consider four types of bacteria: GC, GD, SC, and SD. G = Generous (HGT+), S = Secretive (HGT−), C = Cooperator (produces public goods), and D = Defector (does not produce public goods).

Let $N = [N_{GC},N_{GD},N_{SC},N_{SD}]$ be the population numbers and $x = \frac{N}{N_{total}} = [GC,GD,SC,SD]$ be the population density fractions.

Let the fitness of each type be

$$
\begin{align*}
f_{GC} &= 1 - f_{c} \
f_{GD} &= 1 \
f_{SC} &= 1 + f_{s} - f_{c}\
f_{SD} &= 1 + f_{s}
\end{align*}
$$

Where $f_{s}$ is the benefit of not performing HGT (or the cost of performing HGT), and $f_{c}$ is the cost of cooperation. The benefit is shared among all types.

Let $\alpha$ be the rate of conjugation, such that $\alpha dt$ is the number of conjugation events occurring in the time interval $dt$:

$$
\begin{align}
GC + SD &\xrightarrow{\alpha} GC + SC \
GC + GD &\xrightarrow{\alpha} 2GC
\end{align}
$$

Now we can write down the ODEs assuming large $N$,

$$
\begin{align}
\dot{GC} &= (f_{GC} - \bar{f})GC + \alpha.GC.GD \
\dot{GD} &= (f_{GD} - \bar{f})GD - \alpha.GC.GD \
\dot{SC} &= (f_{SC} - \bar{f})SC + \alpha.GC.SD \
\dot{SD} &= (f_{SD} - \bar{f})SD - \alpha.GC.SD
\end{align}
$$

Where $\bar{f}$ is the average fitness of the population, given by

$$
\begin{align}
\bar{f} &= GC(1 - f_{c}) + GD + SC(1 + f_{s} - f_{c}) + SD(1 + f_{s}) \
&= (GC + GD + SC + SD) - f_{c}(GC+SC) + f_{s}(SC+SD) \
&= 1 - f_{c}(GC+SC) + f_{s}(SC+SD)
\end{align}
$$

Simplifying the ODEs gives us

$$
\begin{align}
\dot{GC} &= GC[- f_{c} + f_{c}(GC+SC) - f_{s}(SC+SD) + \alpha.GD] \
\dot{GD} &= GD[f_{c}(GC+SC) - f_{s}(SC+SD) - \alpha.GC] \
\dot{SC} &= SC[(f_{s} - f_{c}) + f_{c}(GC+SC) - f_{s}(SC+SD)] + \alpha.GC.SD \
\dot{SD} &= SD[f_{s} + f_{c}(GC+SC) - f_{s}(SC+SD) - \alpha GC]
\end{align}
$$

In this case, SD completely dominates the population for all parameter values.

## Plasmid–Chromosome Synergy

Assuming that the type containing HGT chromosomal genes and the cooperative plasmid synergize with each other and confer some fitness benefit to the host, we can write

$$
\begin{align*}
f_{GC} &= 1 - f_{c} + f_{pc} \
f_{GD} &= 1 \
f_{SC} &= 1 + f_{s} - f_{c}\
f_{SD} &= 1 + f_{s}
\end{align*}
$$

Now we can rewrite the above ODEs as

$$
\begin{align}
\dot{GC} &= (f_{GC} - \bar{f})GC + \alpha.GC.GD \
\dot{GD} &= (f_{GD} - \bar{f})GD - \alpha.GC.GD \
\dot{SC} &= (f_{SC} - \bar{f})SC + \alpha.GC.SD \
\dot{SD} &= (f_{SD} - \bar{f})SD - \alpha.GC.SD
\end{align}
$$

Where $\bar{f}$ is the average fitness of the population, given by

$$
\begin{align}
\bar{f} &= GC(1 - f_{c} + f_{pc}) + GD + SC(1 + f_{s} - f_{c}) + SD(1 + f_{s}) \
&= (GC + GD + SC + SD) + f_{pc}GC - f_{c}(GC+SC) + f_{s}(SC+SD) \
&= 1 + f_{pc}GC - f_{c}(GC+SC) + f_{s}(SC+SD)
\end{align}
$$

Simplifying the ODEs gives us

$$
\begin{align}
\dot{GC} &= GC[f_{pc}(1-GC)- f_{c} + f_{c}(GC+SC) - f_{s}(SC+SD) + \alpha.GD] \
\dot{GD} &= GD[-f_{pc}GC + f_{c}(GC+SC) - f_{s}(SC+SD) - \alpha.GC] \
\dot{SC} &= SC[-f_{pc}GC +(f_{s} - f_{c}) + f_{c}(GC+SC) - f_{s}(SC+SD)] + \alpha.GC.SD \
\dot{SD} &= SD[-f_{pc}GC + f_{s} + f_{c}(GC+SC) - f_{s}(SC+SD) - \alpha GC]
\end{align}
$$

## Differential Plasmid Gain Benefit

Assume that a subset of generous cells contains a plasmid with higher fitness compared to cells not containing this plasmid, and that this plasmid can be transferred through HGT, conferring the same fitness benefit to the recipient.

Now we have eight bacterial types:

$$
x = [GC_{+},GC_{-},GD_{+},GD_{-},SC_{+},SC_{-},SD_{+},SD_{-}]
$$

The fitness of each type can now be written as

$$

$$

