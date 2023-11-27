import { world } from './world.js'
import * as THREE from 'three'
import * as TWEEN from 'tween'
import { update_physics } from './physics.js'

function update(delta) {

	update_physics(delta)
	TWEEN.update()

	const { orbit_cam, orbit } = world

	// zooming
	if (world.controls.zoom_in) {
		orbit_cam.zoom += 0.1
		orbit_cam.updateProjectionMatrix()
		world.controls.zoom_in = false
	}
	if (world.controls.zoom_out) {
		orbit_cam.zoom -= 0.1
		orbit_cam.updateProjectionMatrix()
		world.controls.zoom_out = false
	}

	orbit.update()

}

function action() {
	const clock = new THREE.Clock()
	let lastElapsedTime = 0
	function tick() {
		requestAnimationFrame(tick)
		const { renderer, scene, camera, css_renderer } = world

		const elapsedTime = clock.getElapsedTime()
		const delta = elapsedTime - lastElapsedTime
		lastElapsedTime = elapsedTime

		update(delta)

		renderer.render(scene, camera)
		css_renderer.render(scene, camera)
	}
	tick()
}

export { action }