import torch
from stable_baselines3 import PPO


def evaluate_rl_agent(ppo_agent, env, num_episodes=100):
    """
    Evaluates the trained PPO agent on the environment.
    """
    correct_predictions = 0
    total_frames_used = 0
    total_possible_frames = 0

    print(f"--- Evaluating Agent over {num_episodes} Videos ---")

    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False

        #model will manually track actions to be absolutely foolproof
        frames_kept = 0

        while not done:
            # Set deterministic=False. PPO policies can collapse to a single
            # action (like skipping) if forced to be deterministic too early.
            action, _states = ppo_agent.predict(obs, deterministic=False)

            # Track the agent's action in real-time
            if action == 1:
                frames_kept += 1

            obs, reward, done, truncated, info = env.step(action)

        # Safely extract accuracy (if it skipped everything, it defaults to 0)
        accuracy = info.get('accuracy', 0)

        correct_predictions += accuracy
        total_frames_used += frames_kept
        total_possible_frames += env.max_frames

        #Print the first 5 episodes to see the agent's behavior
        if episode < 5:
            print(f"Video {episode+1}: Accuracy={accuracy}, Frames Used={frames_kept}/{env.max_frames}")

    # Calculate final metrics
    final_accuracy = (correct_predictions / num_episodes) * 100
    avg_frames_used = total_frames_used / num_episodes
    frame_reduction_percent = (1 - (total_frames_used / total_possible_frames)) * 100

    print("\n====================================")
    print("      FINAL EVALUATION RESULTS      ")
    print("====================================")
    print(f"Accuracy:              {final_accuracy:.2f}%")
    print(f"Average Frames Used:   {avg_frames_used:.1f} / {env.max_frames}")
    print(f"Computational Savings: {frame_reduction_percent:.2f}% Frame Reduction")
    print("====================================")

