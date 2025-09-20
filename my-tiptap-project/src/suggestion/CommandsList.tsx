import React, { Component } from "react";
import { Editor } from "@tiptap/react";

export interface CommandItem {
  title: string;
  element?: React.ReactNode;
  command: ({ editor, range }: { editor: Editor; range: any }) => void;
}

interface CommandListProps {
  items: CommandItem[];
  command: (item: CommandItem) => void;
}

interface CommandListState {
  selectedIndex: number;
}

class CommandList extends Component<CommandListProps, CommandListState> {
  state: CommandListState = {
    selectedIndex: 0
  };

  componentDidUpdate(oldProps: CommandListProps) {
    if (this.props.items !== oldProps.items) {
      this.setState({
        selectedIndex: 0
      });
    }
  }

  // Make this method public for external access
  public onKeyDown = ({ event }: { event: KeyboardEvent }): boolean => {
    if (event.key === "ArrowUp") {
      this.upHandler();
      return true;
    }

    if (event.key === "ArrowDown") {
      this.downHandler();
      return true;
    }

    if (event.key === "Enter") {
      this.enterHandler();
      return true;
    }

    return false;
  };

  private upHandler(): void {
    this.setState({
      selectedIndex:
        (this.state.selectedIndex + this.props.items.length - 1) %
        this.props.items.length
    });
  }

  private downHandler(): void {
    this.setState({
      selectedIndex: (this.state.selectedIndex + 1) % this.props.items.length
    });
  }

  private enterHandler(): void {
    this.selectItem(this.state.selectedIndex);
  }

  private selectItem(index: number): void {
    const item = this.props.items[index];

    if (item) {
      this.props.command(item);
    }
  }

  render(): React.ReactNode {
    const { items } = this.props;
    return (
      <div className="items">
        {items.map((item, index) => {
          return (
            <button
              className={`item ${
                index === this.state.selectedIndex ? "is-selected" : ""
              }`}
              key={index}
              onClick={() => this.selectItem(index)}
            >
              {item.element || item.title}
            </button>
          );
        })}
      </div>
    );
  }
}

export default CommandList;