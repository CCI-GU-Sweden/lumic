# Hull dimensions as the number of signatures increases

60 noiseless pixels including one pure pixel per signature; synthetic Gaussian spectra sampled by the approximate 980_32ch detector. Each case has a 8 s limit including process startup.

| Signatures | Matrix rank | npc | Hull dimensions | Outcome | Vertices |
| ---: | ---: | ---: | ---: | --- | ---: |
| 3 | 3 | 3 | 2 | completed | 3 |
| 3 | 3 | 4 | 3 | qhull_error |  |
| 4 | 4 | 4 | 3 | completed | 4 |
| 4 | 4 | 5 | 4 | qhull_error |  |
| 6 | 6 | 6 | 5 | completed | 6 |
| 6 | 6 | 7 | 6 | qhull_error |  |
| 8 | 8 | 8 | 7 | completed | 8 |
| 8 | 8 | 9 | 8 | qhull_error |  |
| 10 | 10 | 10 | 9 | completed | 10 |
| 10 | 10 | 11 | 10 | qhull_error |  |

The well-separated synthetic spectra and pure pixels form an idealized case. A timeout does not prove that a hull is impossible. This tests hull selection, not recovery accuracy or how many fluorophores can be separated in an acquisition.
