from gym.envs.registration import registry, register, make, spec

xmls = [
'cheetah_2_back.xml',
'cheetah_4_allfront.xml',
'cheetah_6_back.xml',
'humanoid_2d_7_left_arm.xml',
'humanoid_2d_8_right_knee.xml',
'nervenet_ostrich.xml',
'walker_4_main.xml',
'walker_7_main.xml',
'cheetah_2_front.xml',
'cheetah_4_back.xml',
'cheetah_6_front.xml',
'humanoid_2d_7_left_leg.xml',
'humanoid_2d_9_full.xml',
'cheetah_3_back.xml',
'cheetah_4_front.xml',
'cheetah_7_full.xml',
'humanoid_2d_7_lower_arms.xml',
'nervenet_fullcheetah.xml',
'walker_2_main.xml',
'walker_5_main.xml',
'cheetah_3_balanced.xml',
'cheetah_5_back.xml',
'hopper_3.xml',
'humanoid_2d_7_right_arm.xml',
'nervenet_halfcheetah.xml',
'cheetah_3_front.xml',
'cheetah_5_balanced.xml',
'hopper_4.xml',
'humanoid_2d_7_right_leg.xml',
'nervenet_halfhumanoid.xml',
'walker_3_main.xml',
'walker_6_main.xml',
'cheetah_4_allback.xml',
'cheetah_5_front.xml',
'hopper_5.xml',
'humanoid_2d_8_left_knee.xml',
'nervenet_hopper.xml',
]
for xml in xmls:
    env_name = xml[:-4]
    register(
        id='%s-v0' % env_name,
        entry_point='gym.envs.environments.%s:ModularEnv' % env_name,
        max_episode_steps=1000,
        kwargs={'xml': '/username/swat/modular-rl/src/environments/xmls/%s' % xml}
        )