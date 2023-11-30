import { GUI } from 'GUI'
import { world } from './world.js'
import { reset_orbit_cam } from './controls.js'
import { load_model_name } from './plot.js'
import { add_ball, clear_balls } from './physics.js'

const gui_items = {
	orbit_camera: () => {
		world.camera = world.orbit_cam
	},
	reset_orbit_camera: () => {
		reset_orbit_cam()
	},
	add_ball: () => {
		add_ball()
	},
	clear_balls: () => {
		clear_balls()
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

	// log option
	gui.add(world, 'log_plot')
		.name('Log Plot')
		.onChange(load_model_name)

	// show terrain option
	gui.add(world, 'show_terrain')
		.name('Show Terrain')
		.onChange(() => {
			world.physics.terrainMesh.visible = world.show_terrain
		})

	// show page option
	gui.add(world, 'show_page')
		.name('Show Page')
		.onChange(() => {
			// world.page.visible = world.show_page
			world.page.visible = true
			// fade in/out page
			let time = 0
			let animateID = requestAnimationFrame(animate)
			function animate() {
				time += 0.01
				if (world.show_page) {
					world.page_div.style.opacity = time
				} else {
					world.page_div.style.opacity = 1 - time
				}
				if (time > 1) {
					time = 1
					if(world.show_page) {
						world.page_div.style.opacity = 1
						world.page.visible = true
					}else{
						world.page_div.style.opacity = 0
						world.page.visible = false
					}
					cancelAnimationFrame(animateID)
				} else {
					requestAnimationFrame(animate)
				}
			}
		})

	world.gui = gui
}

export { init_gui, get_pretty_name }