"""Command-line interface functions for user interactions"""

# --------- Local imports ---------#
from utils.logger import logger
logger = logger(__name__)

# --------- Get user input function ---------#
def get_user_choice(item_type: str, available_items: list) -> str | None:
    """
    Function to get user choice

    :param item_type: Type of item
    :param available_items: List of items available
    :return: String of users choice
    """
    print('\n========================================')
    print(f'Available {item_type}(s):')
    for i, item in enumerate(available_items, 1):
        print(f'{i}: {item}')

    choice = input(f"\nSelect {item_type} (1-{len(available_items)}) or enter name: ").strip()

    # Handle numeric choice
    if choice.isdigit():
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(available_items):
            return available_items[choice_idx]
        else:
            print(f"Invalid choice: {choice}")
            return None

    # Handle name choice
    if choice.lower() in [item.lower() for item in available_items]:
        return choice.lower()

    print(f"Invalid choice: {choice}")
    return None

# --------- Get action choice function ---------#
def get_action_choice(has_model: bool) -> str | None:
    """
    Get user's choice for action to perform

    :param has_model: Boolean determining if model exists
    :return: User's choice or None if invalid
    """

    if has_model:
        print('\nModel found!')
        print('What would you like to do?')
        print('1. Train a new model')
        print('2. Evaluate the current model')
        print('3. Run a demo on the current model')

        choice = input('\nSelection action (1-3) or enter name: ').strip()
        actions = ['train', 'evaluate', 'demo']

        if choice.isdigit():
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(actions):
                return actions[choice_idx]

        if choice.lower() in actions:
            return choice.lower()

        print(f'Invalid choice: {choice}')
        return None

    else:
        print('\nNo existing model found.')
        choice = input('Would you like to train a new model? (y/n): ').strip().lower()
        if choice in ['y', 'yes']:
            return 'train'
        else:
            logger.info('Exiting without training.')
            return None

# --------- Follow up function ---------#
def get_follow_up_action() -> str | None:
    """
    Requests if user wants a followup action

    :return: User choice or None if exiting or invalid
    """

    prompt = '\nWould you like to (e)valuate or (d)emo the model or (b)oth? (e/d/b/n): '
    choice = input(prompt).strip().lower()

    if choice in ['e', 'evaluate']:
        return 'evaluate'
    elif choice in ['d', 'demo']:
        return 'demo'
    elif choice in['b', 'both']:
        return 'both'
    elif choice in ['n', 'no']:
        return None
    else:
        logger.warning('Invalid choice, skipping follow up action')
        return None