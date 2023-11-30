import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import { world } from './world.js'
import * as THREE from 'three'
import TWEEN from 'tween'

function init_keys() {
	const handler = flag => event => {
		switch (event.key) {
			case 'w':
			case 'ArrowUp':
				world.controls.up = flag
				break
			case 's':
			case 'ArrowDown':
				world.controls.down = flag
				break
			case 'a':
			case 'ArrowLeft':
				world.controls.left = flag
				break
			case 'd':
			case 'ArrowRight':
				world.controls.right = flag
				break
			case '+':
				world.controls.zoom_in = flag
				break
			case '-':
				world.controls.zoom_out = flag
				break
		}
	}
	document.addEventListener('keydown', handler(true))
	document.addEventListener('keyup', handler(false))
}

function init_orbit() {
	const { orbit_cam, css_renderer } = world
	const orbit = new OrbitControls(orbit_cam, css_renderer.domElement)
	orbit.enableKeys = true
	orbit.enableDamping = true

	// block iframe events when moving
	const blocker = document.getElementById('blocker')
	blocker.style.display = 'none'
	orbit.addEventListener('start', function () {
		blocker.style.display = ''
	})
	orbit.addEventListener('end', function () {
		blocker.style.display = 'none'
	})

	world.orbit = orbit

	load_orbit(orbit_cam, orbit)
	orbit.addEventListener('change', save_orbit(orbit_cam, orbit))
	// orbit.addEventListener('change', function () {
	// 	// move the page to a fixed distance from the camera, facing the camera
	// 	const { page } = world
	// 	if (!page) return

	// 	const targetPosition = orbit_cam.position.clone()
	// 		// position the page in front of the camera
	// 		.add(orbit_cam.getWorldDirection(new THREE.Vector3()).multiplyScalar(7))
	// 		// move the page to the top left
	// 		.add(new THREE.Vector3(-2.5, 2.5, 0))
	// 	const targetQuaternion = orbit_cam.quaternion.clone()

	// 	const transitions = [
	// 		[page.position, targetPosition],
	// 		[page.quaternion, targetQuaternion],
	// 	]

	// 	// smooth transition
	// 	for (const [start, end] of transitions) {
	// 		new TWEEN.Tween(start)
	// 			.to(end, 1000)
	// 			.easing(TWEEN.Easing.Quadratic.InOut)
	// 			.onUpdate(() => {
	// 				orbit_cam.updateProjectionMatrix()
	// 				orbit.update()
	// 			}).start()
	// 	}


	// })
}

function save_orbit(orbit_cam, controls) {
	return () => {
		localStorage.setItem('camera_position', JSON.stringify(orbit_cam.position))
		localStorage.setItem('camera_rotation', JSON.stringify(orbit_cam.rotation))
		localStorage.setItem('controls_target', JSON.stringify(controls.target))
	}
}

function load_orbit(orbit_cam, orbit) {
	const camera_position = JSON.parse(localStorage.getItem('camera_position'))
	const camera_rotation = JSON.parse(localStorage.getItem('camera_rotation'))
	const controls_target = JSON.parse(localStorage.getItem('controls_target'))
	if (!camera_position || !camera_rotation || !controls_target) return

	orbit_cam.position.x = camera_position.x
	orbit_cam.position.y = camera_position.y
	orbit_cam.position.z = camera_position.z
	orbit_cam.rotation._x = camera_rotation._x
	orbit_cam.rotation._y = camera_rotation._y
	orbit_cam.rotation._z = camera_rotation._z
	orbit.target.x = controls_target.x
	orbit.target.y = controls_target.y
	orbit.target.z = controls_target.z
	orbit_cam.updateProjectionMatrix()
	orbit.update()
}


function init_swipes() {
	const hammer = new Hammer(document.body, {
		recognizers: [
			[Hammer.Swipe],
			[Hammer.Tap],
			[Hammer.Tap, { event: 'doubletap', taps: 2 }],
		]
	})
}

function init_cameras() {
	const { scene } = world

	const orbit_cam = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 5000)
	orbit_cam.position.set(0, 2, 3)
	scene.add(orbit_cam)

	world.orbit_cam = orbit_cam
}

function reset_orbit_cam() {
	const { orbit_cam, orbit } = world
	const transitions = [
		[orbit_cam.position, new THREE.Vector3(0.8, 1.2, 1)],
		[orbit_cam.quaternion, new THREE.Quaternion().setFromEuler(new THREE.Euler(0, 0, 0, 'XYZ'))],
		[orbit.target, world.origin],
	]
	// smooth transition
	for (const [start, end] of transitions) {
		new TWEEN.Tween(start)
			.to(end, 1000)
			.easing(TWEEN.Easing.Quadratic.InOut)
			.onUpdate(() => {
				orbit_cam.updateProjectionMatrix()
				orbit.update()
			}).start()
	}
}

export { init_keys, init_swipes, init_cameras, init_orbit, reset_orbit_cam }