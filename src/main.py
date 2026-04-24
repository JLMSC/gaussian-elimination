import numpy as np
import PySimpleGUI as sg
from utils import back_substitution, row_echelon_form, string_to_augmented_matrix

# pyright: reportOptionalMemberAccess=false

__INPUT_KEY: str = '-INPUT-'
__SUBMIT_BTN_KEY: str = '-SUBMIT-'
__CANCEL_BTN_KEY: str = '-CANCEL-'


def gaussian_elimination(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Solve a linear system represented by an augmented matrix using the Gaussian elimination method.

    Args:
        A (np.ndarray): Square matrix of size n x n representing the coefficients of the linear system.
        B (np.ndarray): Column matrix of size n x 1 representing the constant terms.

    Returns:
        np.ndarray: The solution vector.
    """
    row_echelon_M = row_echelon_form(A, B)
    solution = back_substitution(row_echelon_M)
    return solution


def main() -> None:
    """Main function, entrypoint."""

    def show_invalid_linear_system_popup(message: str) -> None:
        """Shows a popup with a message.

        Args:
            message (str): Message to be displayed on the popup.
        """
        sg.popup(message, title='Error')
        window[__INPUT_KEY].update('')  # type: ignore

    layout = [
        [
            sg.Text('Write any linear system:'),
        ],
        [
            sg.Multiline(key=__INPUT_KEY, enable_events=True, size=(70, 8)),
        ],
        [
            sg.Button('Submit', key=__SUBMIT_BTN_KEY, disabled=True),
            sg.Button('Cancel', key=__CANCEL_BTN_KEY, disabled=True),
        ],
    ]
    window = sg.Window(
        'Gaussian Elimination',
        layout,
        resizable=False,
        size=(500, 200),
    )
    while True:
        event, values = window.read()
        if event == sg.WINDOW_CLOSED:
            break
        text = values[__INPUT_KEY]
        if event == __INPUT_KEY:
            has_text = bool(text.strip())
            window[__SUBMIT_BTN_KEY].update(disabled=not has_text)
            window[__CANCEL_BTN_KEY].update(disabled=not has_text)
        if event == __CANCEL_BTN_KEY:
            window[__INPUT_KEY].update('')  # type: ignore
            window[__SUBMIT_BTN_KEY].update(disabled=True)
            window[__CANCEL_BTN_KEY].update(disabled=True)
        if event == __SUBMIT_BTN_KEY:
            variables, A, B = string_to_augmented_matrix(text)
            if variables == 'Invalid linear system' or np.isclose(np.linalg.det(A), 0):
                show_invalid_linear_system_popup('Invalid Linear System')
            else:
                sols = gaussian_elimination(A, B)
                if not isinstance(sols, str):
                    res = '\n'.join(
                        f'{var} = {val:.4f}' for var, val in zip(variables.split(), sols)
                    )
                    sg.popup(f'{res}', title='Result')
                else:
                    sg.popup(f'{sols}', title='Result')


if __name__ == '__main__':
    main()
