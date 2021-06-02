def group_by_normal_interesting(fishes, episodes):
    fishes_groups = {
        "normal": [],
        "interesting": []
    }
    
    for fish in fishes:
        found_episode = False
        for episode in episodes:
            if episode.fish_id == fish.fish_id:
                found_episode = True
                break
        
        if found_episode:
            fishes_groups["interesting"].append(fish)
        
        else:
            fishes_groups["normal"].append(fish)
        
    return fishes_groups
