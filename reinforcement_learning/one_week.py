SEDENTARY_WORK = {"lower": [4, 7.5, 0.5], "upper": [11, 18, 12]}
ACTIVE_WORK = {"lower": [4, 1, 7.5], "upper": [11, 12, 18]}
ACTIVE_WEEKEND = {"lower": [4, 1, 4], "upper": [9, 5, 16]}
RELAXING_WEEKEND = {"lower": [8, 6, 0.5], "upper": [12, 18, 2]}
SOCIAL_WEEKEND = {"lower": [4, 4, 1], "upper": [10, 18, 6]}

week_a = [SEDENTARY_WORK for _ in range(5)] + [ACTIVE_WEEKEND, RELAXING_WEEKEND]
weeb_b = [ACTIVE_WORK for _ in range(5)] + [RELAXING_WEEKEND, SOCIAL_WEEKEND]

for day in week_a:
    print(day)
