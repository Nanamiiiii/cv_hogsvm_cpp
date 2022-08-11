ls | sort Name | % {$i = 1} {mv $_.Name ("{0:000}.jpg" -f $i++)}
