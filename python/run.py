from baseline import main_parallel, main

main_parallel([10, 20, 50], ['PC', 'GES'], 10)
# FIXME PC might stuck at some data. Thus I probably want to set a timeout
main_parallel([100], ['GES', 'PC'], 10)
main_parallel([200], ['GES'], 10)
main_parallel([150], ['PC'], 10)
main_parallel([200], ['PC'], 10)
main_parallel([10, 20, 50], ['CAM'], 10)
# main([10, 20, 50, 100], ['notears'], 5)

main([300, 400], ['GES'], 5)
# Note: run with timeout for PC
main([300, 400], ['PC'], 4)
# CAM d=100 is so slow
# main([100], ['CAM'], 5)