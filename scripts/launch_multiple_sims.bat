set n_iter=%1
set brain=%2
for /l %%x in (1, 1, %n_iter%) do (
	echo %%x
	start python hub.py --brain %brain% --log-iterations %3
	timeout 1
	)