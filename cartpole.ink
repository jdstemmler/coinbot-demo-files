inkling "2.0"

type GameState {
    position: number,
    velocity: number,
    angle: number,
    rotation: number
}

const left = -1
const right = 1

type Action {
    command: number<left, right>
}

simulator the_simulator(action: Action): GameState {
}

graph (input: GameState): Action {

    concept balance(input): Action {
        curriculum {
            source the_simulator
        }
    }
    output balance
}
