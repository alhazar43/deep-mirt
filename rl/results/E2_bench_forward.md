# E2 bench forward results

- cuda_available, True
- cuda_device, NVIDIA GeForce RTX 4060 Laptop GPU
- n_categories, 4
- n_questions, 200
- platform, Windows-10-10.0.26200-SP0
- python_version, 3.11.13
- repeat, 10
- torch_version, 2.7.1+cu126
- warmup_iters, 3

| Encoder | Device | B | T | Median (ms) | Mean (ms) | Min (ms) | Max (ms) | Repeats |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dkvmn | cpu | 1 | 50 | 5.068 | 5.071 | 4.808 | 5.262 | 10 |
| dkvmn | cpu | 128 | 50 | 37.996 | 37.993 | 36.733 | 39.051 | 10 |
| lstm | cpu | 1 | 50 | 4.027 | 4.230 | 3.829 | 6.206 | 10 |
| lstm | cpu | 128 | 50 | 9.029 | 9.067 | 8.866 | 9.710 | 10 |
| transformer | cpu | 1 | 50 | 3.026 | 3.043 | 2.948 | 3.197 | 10 |
| transformer | cpu | 128 | 50 | 13.518 | 13.639 | 13.109 | 14.415 | 10 |
| dkvmn | cuda | 1 | 50 | 19.743 | 19.541 | 18.612 | 20.321 | 10 |
| dkvmn | cuda | 128 | 50 | 20.945 | 20.403 | 18.266 | 21.982 | 10 |
| lstm | cuda | 1 | 50 | 1.098 | 1.166 | 1.049 | 1.662 | 10 |
| lstm | cuda | 128 | 50 | 1.503 | 1.505 | 1.475 | 1.556 | 10 |
| transformer | cuda | 1 | 50 | 2.626 | 2.669 | 2.507 | 3.121 | 10 |
| transformer | cuda | 128 | 50 | 3.027 | 3.031 | 2.985 | 3.134 | 10 |
