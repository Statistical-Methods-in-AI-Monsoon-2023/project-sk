import { world } from './world.js'
import * as THREE from 'three'
import * as TWEEN from 'tween'

function update(delta) {
	
	TWEEN.update()

}

function action() {
	const clock = new THREE.Clock()
	let lastElapsedTime = 0
	function tick() {
		requestAnimationFrame(tick)
		const { renderer, scene, camera } = world

		const elapsedTime = clock.getElapsedTime()
		const delta = elapsedTime - lastElapsedTime
		lastElapsedTime = elapsedTime

		update(delta)

		renderer.render(scene, camera)
	}
	tick()
}

export { action }