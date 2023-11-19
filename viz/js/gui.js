import { GUI } from 'GUI'
import { world } from './world.js'
import { reset_orbit_cam } from './controls.js'
import { load_model_name } from './plot.js'

const gui_items = {
	orbit_camera: () => {
		world.camera = world.orbit_cam
	},
	reset_orbit_camera: () => {
		reset_orbit_cam()
	},
}

function get_pretty_name(name) {
	// remove underscores and capitalize each word
	return name.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')
}

function init_gui() {
	const gui = new GUI()

	for (const item in gui_items) {
		gui.add(gui_items, item).name(get_pretty_name(item))
	}

	// select model option
	gui.add(world, 'active_model', world.model_names)
		.name('Choose Model')
		.onChange(load_model_name)

	world.gui = gui
}

export { init_gui, get_pretty_name }