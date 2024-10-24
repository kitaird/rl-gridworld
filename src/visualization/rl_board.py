import random
import numpy as np
from collections import UserList
from tkinter import Tk, Canvas, NW, StringVar, OptionMenu, TOP, Frame, LEFT, Button, Label

import game2dboard
import scipy.stats as stats

from src.env.action import Action
from src.env.state import State


def compute_color_for_value(data, value):
    normalized_value = stats.percentileofscore(data, value)/100
    blue = 255 * (1 - normalized_value)
    red = 255 * normalized_value
    return f'#{int(red):02x}{0:02x}{int(blue):02x}'


def gridworld_parser(val: State):
    if val.wall:
        return 'wall'
    if val.reward > 0:
        return f'+{val.reward}'
    if val.reward < 0:
        return f'{val.reward}'
    if val.terminal:
        return '0'
    return 'field'


class RlBoard(UserList):
    """
    A graphical user interface for 2d arrays (matrix)
    """

    def __init__(
            self,
            agents,
            environment):
        """

        Creates an App

        :param int nrows:
            The number of rows.

        :param int ncols:
            The number of columns.
        """

        UserList.__init__(self)             # Initialize parent class
        # Create list [ncols][nrows]
        nrows = environment.rows
        ncols = environment.cols
        self.extend([self._BoardRow(ncols, self) for _ in range(nrows)])

        self._agent = None
        self._agents = {a.algo_name: a for a in agents}
        self._nrows = nrows
        self._ncols = ncols
        self._isrunning = False
        self._states = environment.all_states
        self._env = environment
        # Array used to store cells elements (rectangles)
        self._cells = [[None] * ncols for _ in range(nrows)]

        # The window
        self._root = Tk()
        # cell's container
        self._canvas = Canvas(self._root, highlightthickness=0)
        self._background_image = None           # background image file name
        # rectange for grid color
        self._bgrect = self._canvas.create_rectangle(1, 1, 2, 2, width=0)
        self._bgimage_id = None
        self._msgbar = None                 # message bar component

        # Fields for board properties
        self._title = "game2dboard"             # default window title
        self._cursor = "arrow"                  # default mouse cursor
        self._margin = 5                        # default board margin (px)
        # default grid cell_spacing (px)
        self._cell_spacing = 1
        self._margin_color = "light grey"       # default border color
        self._cell_color = "white"              # default cell color
        self._grid_color = "black"              # default grid color

        self._on_start = None                   # game started callback
        self._on_key_press = None               # user key press callback
        self._on_mouse_click = None             # user mouse callback
        self._on_timer = None                   # user timer callback

        # event
        self._timer_interval = 0            # ms
        self._after_id = None               # current timer id
        self._is_in_timer_calback = False
        # register internal key callback
        self._root.bind("<Key>", self._key_press_clbk)
        # register internal mouse callback
        self._canvas.bind("<ButtonPress>", self._mouse_click_clbk)
        if self._states:
            self.show_grid()
        self._command_stack = []
        self._last_show_command = None
        self.on_mouse_click = self.print_reward_for_cell
        self.cell_size = 100
        self.cell_color = "white"
        self._warning_label = None
        self.init_board_title()

    def print_reward_for_cell(self, btn, row, col):
        print("Data for row[" + row.__str__() + "] col[" + col.__str__() + "] is " + self[row][col].__str__())

    def init_board_title(self):
        if self.agent is None:
            self.title = "Grid world game"
        else:
            self.title = "Grid world game - Current agent: " + self.agent.algo_name

    def __getitem__(self, row):             # subscript getter: self[row]
        # Store last accessed row (NOT thread safe... )
        self._BoardRow.current_i = row
        return super().__getitem__(row)     # return a _BoardRow

    # Properties
    # ---------------------------------------------------------------

    @property
    def size(self):
        """

        Number of elements in the array (readonly).

        :type: int
        """
        return self._nrows * self._ncols

    @property
    def nrows(self):
        """

        Number of rows in the array (readonly).

        :type: int
        """
        return self._nrows

    @property
    def ncols(self):
        """

        Number of columns in the array (readonly).

        :type: int
        """
        return self._ncols

    @property
    def width(self):
        """

        Board width, in px. Only available after .show() (readonly).

        :type: int
        """
        return self._root.winfo_reqwidth()

    @property
    def height(self):
        """

        Board height, in px. Only available after .show() (readonly).

        :type: int
        """
        return self._root.winfo_reqheight()

    @property
    def title(self):
        """

        Gets or sets the window title.

        :type: str
        """
        return self._title

    @title.setter
    def title(self, value):
        self._title = value
        self._root.title(value)

    @property
    def agent(self):
        return self._agent

    @property
    def env(self):
        return self._env

    def fill_field(self, row, col, value):
        self[row][col] = value

    def show_grid(self):
        self.load(self._states)

    @property
    def cursor(self):
        """

        Gets or sets the mouse cursor shape.<br>
        Setting to None hides the cursor.

        :type: str
        """
        return self._cursor

    @cursor.setter
    def cursor(self, value):
        self._cursor = value
        if value is None:
            value = "none"
        self._canvas.configure(cursor=value)

    @property
    def margin(self):
        """

        Gets or sets the board margin (px).

        :type: int
        """
        return self._margin

    @margin.setter
    def margin(self, value):
        if self._isrunning:
            raise Exception("Can't update margin after show()")
        self._margin = value

    @property
    def cell_spacing(self):
        """
        Gets or sets the space between cells (px).

        :type: int
        """
        return self._cell_spacing

    @cell_spacing.setter
    def cell_spacing(self, value):
        if self._isrunning:
            raise Exception("Can't update cell_spacing after show()")
        self._cell_spacing = value

    @property
    def margin_color(self):
        """
        Gets or sets the margin_color.

        :type: str
        """
        return self._margin_color

    @margin_color.setter
    def margin_color(self, value):
        self._margin_color = value
        self._canvas.configure(bg=value)

    @property
    def cell_color(self):
        """
        Gets or sets cells color

        :type: str
        """
        return self._cell_color

    @cell_color.setter
    def cell_color(self, value):
        self._cell_color = value
        # Update bgcolor for all cells
        if self._isrunning:
            for row in self._cells:
                for cell in row:
                    cell.bgcolor = value

    @property
    def grid_color(self):
        """
        Gets or sets grid color

        :type: str
        """
        return self._grid_color

    @grid_color.setter
    def grid_color(self, value):
        self._grid_color = value
        self._canvas.itemconfig(
            self._bgrect, fill=value if value is not None else '')

    @property
    def cell_size(self):
        """
        Gets or sets the cells dimension (width, height)

        :type: int or (int, int)
        """
        return game2dboard.Cell.size

    @cell_size.setter
    def cell_size(self, value):
        if self._isrunning:
            raise Exception("Can't resize cells after run()")
        # size is a tuple (width, height)
        if type(value) is not tuple:
            v = int(value)
            value = (v, v)
        # All cells has same size (class field)
        game2dboard.Cell.size = value

    @property
    def background_image(self):
        """
        Gets or sets the board's background image


        :type: str
        """
        return self._background_image

    @background_image.setter
    def background_image(self, value):
        if self._background_image != value:
            self._background_image = value
            if self._bgimage_id:
                self._canvas.delete(self._bgimage_id)    # clear current image
            if value is not None:
                self.grid_color = self.margin_color = self.cell_color = None
                image = game2dboard.ImageMap.get_instance().load(value)
                if image is not None:
                    self._image_object = image
                    self._bgimage_id = self._canvas.create_image(  # Draw a image
                        0,
                        0,
                        anchor=NW,
                        image=image)
                    self._canvas.tag_lower(self._bgimage_id)

    # Private properties
    @property
    def _canvas_width(self):
        return self._ncols * (game2dboard.Cell.width + self.cell_spacing) - self.cell_spacing + (2 * self.margin)

    @property
    def _canvas_height(self):
        return self._nrows * (game2dboard.Cell.height + self.cell_spacing) - self.cell_spacing + (2 * self.margin)

    # Events
    # ---------------------------------------------------------------

    # Game state events
    @property
    def on_start(self):
        """
        Gets or sets the game started callback function.
        The GUI is ready and the program is going to enter the main loop.

        :type: function()
        """
        return self._on_start

    @on_start.setter
    def on_start(self, value):
        self._on_start = value

    # Keyboard events
    @property
    def on_key_press(self):
        """
        Gets or sets the keyboard callback function

        :type: function(key: str)
        """
        return self._on_key_press

    @on_key_press.setter
    def on_key_press(self, value):
        self._on_key_press = value

    # Internal callback
    def _key_press_clbk(self, ev):
        if callable(self._on_key_press):
            self._on_key_press(ev.keysym)

    # Mouse click events
    @property
    def on_mouse_click(self):
        """
        Gets or sets the mouse callback function

        :type: function(button: str, row: int, col: int)
        """
        return self._on_mouse_click

    @on_mouse_click.setter
    def on_mouse_click(self, value):
        self._on_mouse_click = value

    # Internal callback
    def _mouse_click_clbk(self, ev):
        if callable(self._on_mouse_click):
            rc = self._xy2rc(ev.x, ev.y)
            if rc:
                self._on_mouse_click(ev.num, rc[0], rc[1])

    # Timer events
    @property
    def on_timer(self):
        """
        Gets or sets the timer callback function

        :type: function
        """
        return self._on_timer

    @on_timer.setter
    def on_timer(self, value):
        self._on_timer = value

    # internal callback
    def _timer_clbk(self):
        if self._timer_interval > 0 and callable(self._on_timer):
            self._is_in_timer_calback = True
            self._on_timer()              # Call the user callback function
            self._is_in_timer_calback = False
        if self._timer_interval > 0:      # User callback function may change timer!
            self._after_id = self._root.after(
                self._timer_interval, self._timer_clbk)

    # Methods
    # ---------------------------------------------------------------

    def show(self):
        """

        Create the GUI, display and enter the run loop.

        """
        self._setupUI()
        self._isrunning = True
        if callable(self._on_start):
            self._on_start()
        self._run_command_stack()
        self._root.mainloop()

    def _run_command_stack(self):
        for cmd in self._command_stack:
            cmd()

    def clear(self):
        """

        Clear the board, setting all values to None.
        """
        self.fill(None)

    def close(self):
        """

        Close the board, exiting the program.
        """
        self._root.quit()

    def create_output(self, **kwargs):
        """

        Create a output message bar.
        kwargs:
            color = str
            background_color` = str
            font_size = int
        """
        if self._isrunning:
            raise Exception("Can't create output after run()")
        elif self._msgbar is None:
            self._msgbar = game2dboard.OutputBar(
                self._root, **kwargs)

    def print(self, *objects, sep=' ', end=''):
        """

        Print message to output bar.
        Use like standard print() function.
        """
        if self._msgbar:
            s = sep.join(str(obj) for obj in objects) + end
            self._msgbar.show(s)

    def shuffle(self):
        """

        Random shuffle all values in the board
        """

        # Copy all values to an array, random.shuffle it, then copy back
        a = []
        for r in self:
            a.extend(r)
        random.shuffle(a)
        for row in self:
            for c in range(self._ncols):
                row[c] = a.pop()

    def fill(self, value, row=None, col=None):
        """

        Fill the board (or a row, or a column) with a value

        :param value: The value to store
        :param int row: Index of row to fill. Default=None (all rows)
        :param int col: Index of column to fill. Default=None (all columns)
        """
        if row is None and col is None:         # all rows and columns
            for r in range(self._nrows):
                for c in range(self._ncols):
                    self[r][c] = value
        elif row is not None and col is None:  # single row
            for c in range(self._ncols):
                self[row][c] = value
        elif row is None and col is not None:   # a single column
            for r in range(self._nrows):
                self[r][col] = value
        else:
            raise Exception("Invalid argument supplied (row= AND col=)")

    def copy(self):
        """
        Returns a shallow copy of the array (only data, not the GUI) into a regular Python list (of lists).
        """
        return [[self[i][j] for j in range(self.ncols)] for i in range(self.nrows)]

    def load(self, data):
        """
        Copy data from regular Python 2D array (list of lists) into the Board.
        """
        if len(data) < self._nrows:
            raise IndexError()
        for r in range(self._nrows):
            if len(data) != self._ncols * self._nrows:
                raise IndexError()
            for c in range(self._ncols):
                if self.env.start_state is not None and self.env.start_state.row == r and self.env.start_state.col == c:
                    self[r][c] = 'agent'
                else:
                    self[r][c] = gridworld_parser(data[(r,c)])

    def start_timer(self, msecs):
        """

        Start a periodic timer that executes the a function every msecs milliseconds

        The callback function must be registered using .on_timer property.

        :param int msecs: Time in milliseconds.
        """
        if msecs != self._timer_interval:                       # changed
            self.stop_timer()
            self._timer_interval = msecs
            if msecs > 0 and not self._is_in_timer_calback:
                self._after_id = self._root.after(msecs, self._timer_clbk)

    def stop_timer(self):
        """

        Stops the current timer.
        """
        self._timer_interval = 0
        if self._after_id:
            self._root.after_cancel(self._after_id)
            self._after_id = None

    def pause(self, msecs, change_cursor=True):
        """

        Delay the program execution for a given number of milliseconds.

        Warning: long pause freezes the GUI!

        :param int msecs: Time in milliseconds.
        :param bool change_cursor: Change the cursor to "watch" during pause?
        """
        if change_cursor:
            oldc = self.cursor
            self.cursor = "watch"
        self._root.update_idletasks()
        self._root.after(msecs)
        if change_cursor:
            self.cursor = oldc

    # Private methods
    # ---------------------------------------------------------------

    def _setupUI(self):
        # Window is not resizable
        self._root.resizable(False, False)
        self.background_image = self._background_image  # Draw background image
        if self._background_image is not None:
            self.margin_color = self.grid_color = self.cell_color = None
        else:
            self.margin_color = self._margin_color          # Paint background
            self.grid_color = self._grid_color              # Table internal lines
            self.cell_color = self._cell_color              # Cells background

        self.margin = self._margin                      # Change root's margin
        self.cell_spacing = self._cell_spacing          # Change root's padx/y
        self.title = self._title                        # Update window's title
        self.cursor = self._cursor
        self._resize_canvas()
        # Create all cells
        for r in range(self._nrows):
            for c in range(self._ncols):
                x, y = self._rc2xy(r, c)
                newcell = game2dboard.Cell(self._canvas, x, y)
                newcell.bgcolor = self._cell_color
                self._cells[r][c] = newcell
                if self[r][c] != None:                       # Cell has a value
                    self._notify_change(r, c, self[r][c])    # show it

        self._setup_buttons()
        self._canvas.pack()
        self._root.update()

    def _setup_buttons(self):
        row_frame = Frame(self._root)
        row_frame.pack(side=TOP)

        var = StringVar(row_frame)
        algo_names = list(self._agents.keys())
        var.set("Select an agent")
        drop_down = OptionMenu(row_frame, var, *algo_names, command=self.set_agent)
        drop_down.pack()

        iterate_btn = Button(row_frame, text="Iterate", command=self.iterate_action)
        iterate_btn.pack(side=LEFT)

        grid_btn = Button(row_frame, text="Show grid", command=lambda: self.show_wrapper(self.show_grid))
        grid_btn.pack(side=LEFT)

        loss_btn = Button(row_frame, text="Show values", command=lambda: self.show_wrapper(self.show_values))
        loss_btn.pack(side=LEFT)

        gradient_btn = Button(row_frame, text="Show policy", command=lambda: self.show_wrapper(self.show_policy))
        gradient_btn.pack(side=LEFT)

        reset_btn = Button(row_frame, text="Clear", command=self.clear_action)
        reset_btn.pack(side=LEFT)

    def set_agent(self, selected_algo_name):
        self._agent = self._agents[selected_algo_name]
        self.init_board_title()
        if self._warning_label is not None:
            self._warning_label.destroy()
            self._warning_label = None

    def clear_action(self):
        if self.agent is not None:
            self._agent.clear()
            if self._last_show_command is None:
                self._last_show_command = self.show_grid
            self._last_show_command()
        else:
            self.show_alert_agent_is_none()

    def iterate_action(self):
        if self.agent is not None:
            self._agent.run()
            if self._last_show_command is None:
                self._last_show_command = self.show_grid
            self._last_show_command()
        else:
            self.show_alert_agent_is_none()

    def show_wrapper(self, show_command):
        if self.agent is not None:
            self._last_show_command = show_command
            self.show_grid()
            show_command()
        else:
            self.show_alert_agent_is_none()

    def show_alert_agent_is_none(self):
        var = StringVar()
        var.set("Agent not yet selected! Select an Agent first!")
        if self._warning_label is None:
            self._warning_label = Label(self._root, fg='red', textvariable=var)
            self._warning_label.pack()

    def show_values(self):
        if hasattr(self._agent, 'state_values'):
            self.show_state_values()
        else:
            self.show_action_values()

    def show_state_values(self):
        min_abs_value = np.abs(min(self._agent.state_values.all_values()))
        abs_values = self._agent.state_values.all_values() + min_abs_value
        for state, state_value in self._agent.state_values._values.items():
            if state_value is not None:
                self.fill_field(state.row, state.col, "{:1.3f}".format(state_value))
                self._cells[state.row][state.col].bgcolor = compute_color_for_value(abs_values, state_value + min_abs_value)

    def show_action_values(self):
        min_abs_value = np.abs(min(self._agent.action_values.all_values()))
        abs_values = self._agent.action_values.all_values() + min_abs_value
        for state, action_value_dict in self._agent.action_values._values.items():
            if action_value_dict is not None:
                action_values_string = self.get_formatted_action_value_string(action_value_dict)
                self.fill_field(state.row, state.col, action_values_string)
                action_probabilities = self._agent.policy[state]
                weighted_action_value = sum([action_probabilities[a] * action_value_dict[a] for a in self._env.actions]) + min_abs_value
                self._cells[state.row][state.col].bgcolor = compute_color_for_value(abs_values, weighted_action_value)

    def get_formatted_action_value_string(self, action_values):
        num_of_actions = len(self._env.actions)
        if num_of_actions == 4:
            return "|     {:1.2f}     |\n{:1.2f} | {:1.2f}\n|     {:1.2f}     |".format(action_values[Action.N], action_values[Action.W], action_values[Action.E], action_values[Action.S])
        elif num_of_actions == 8:
            return "{:1.1f} \\ {:1.1f} / {:1.1f}\n{:1.1f} |        | {:1.1f}\n{:1.1f} / {:1.1f} \\ {:1.1f}".format(action_values[Action.NW],action_values[Action.N],action_values[Action.NE], action_values[Action.W], action_values[Action.E], action_values[Action.SW], action_values[Action.S], action_values[Action.SE])
        else:
            raise ValueError("Invalid number of actions")

    def show_policy(self):
        for state, action_probs in self._agent.policy.state_action_probabilities.items():
            action = max(self.env.actions, key=lambda a: action_probs[a])
            self.fill_field(state.row, state.col, action.name)

    def _notify_change(self, row, col, new_value):
        if self._cells[row][col] is not None:
            self._cells[row][col].value = new_value

    # Config the canvas size
    def _resize_canvas(self):
        self._canvas.config(width=self._canvas_width, height=self._canvas_height)

        x1 = y1 = self.margin
        x2 = self._canvas_width - x1
        y2 = self._canvas_height - y1
        self._canvas.coords(self._bgrect, x1, y1, x2, y2)

    # Translate [row][col] to canvas coordinates
    def _rc2xy(self, row, col):
        x = col * (game2dboard.Cell.width + self.cell_spacing) + self.margin
        y = row * (game2dboard.Cell.height + self.cell_spacing) + self.margin
        return (x, y)

    # Translate canvas coordinates to (row, col)
    def _xy2rc(self, x, y):
        # how can i optimize it ???? May be _self.canvas.find_withtag(CURRENT)
        for r in range(self._nrows):
            for c in range(self._ncols):
                cell = self._cells[r][c]
                if cell.x < x < cell.x + game2dboard.Cell.width \
                        and cell.y < y < cell.y + game2dboard.Cell.height:
                    return (r, c)
        return None

    # Inner classes
    # ---------------------------------------------------------------

    # A row is a list, so I can use the magic function __setitem__(board[i][j])

    class _BoardRow(UserList):
        # Last acessed row (class member).
        # Yes, its not thread safe!
        # Maybe in the future I will use a proxy class
        current_i = None

        def __init__(self, length, parent):
            UserList.__init__(self)
            self.extend([None] * length)         # Initialize the row
            self._parent = parent           # the board

        def __setitem__(self, j, value):
            self._parent._notify_change(self.__class__.current_i, j, value)
            return super().__setitem__(j, value)
