def profile_it(profile_filename, f, *args):
    import hotshot
    prof = hotshot.Profile(profile_filename)
    print 'Profiling %s' % f.func_name
    prof.runcall(f, *args)
    prof.close()

def print_profile_data(filename):
    import hotshot.stats
    stats = hotshot.stats.load(filename)
    stats.strip_dirs()
    stats.sort_stats('time', 'calls')
    stats.print_stats()
    print '--Callers--'
    stats.print_callers()
    print '--Callees--'
    stats.print_callees()