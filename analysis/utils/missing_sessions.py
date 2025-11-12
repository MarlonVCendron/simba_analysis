from analysis.consts import ALL_SESSIONS

def print_session_summary(dlc_data):
    sorted_rat_ids = sorted(dlc_data.keys())
    
    for rat_id in sorted_rat_ids:
        rat_sessions = dlc_data[rat_id]
        available_sessions = sorted([sess for sess in rat_sessions.keys()])
        missing_sessions = sorted([sess for sess in ALL_SESSIONS if sess not in rat_sessions])
        
        available_str = ', '.join(available_sessions) if available_sessions else 'None'
        missing_str = ', '.join(missing_sessions) if missing_sessions else 'None'
        
        print(f"{rat_id:<12} {available_str:<30} {missing_str:<30}")

def print_complete_rats(dlc_data, session=None):
    sorted_rat_ids = sorted(dlc_data.keys())
    
    complete_rats = []
    for rat_id in sorted_rat_ids:
        rat_sessions = dlc_data[rat_id]
        if len(rat_sessions) == len(ALL_SESSIONS):
            complete_rats.append(rat_id)
    
    print(f"Ratos com todas as sessões: {complete_rats}")