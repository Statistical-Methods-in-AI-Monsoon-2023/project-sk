import * as THREE from 'three'
import { world } from './world.js'

function resize_callback() {
	const { camera, renderer } = world
	renderer.setSize(window.innerWidth, window.innerHeight)
	camera.aspect = window.innerWidth / window.innerHeight
	camera.updateProjectionMatrix()
}

function init_canvas() {
	window.addEventListener('resize', resize_callback)

	const renderer = new THREE.WebGLRenderer({
		antialias: true,
	})
	document.body.appendChild(renderer.domElement)
	
	renderer.setSize(window.innerWidth, window.innerHeight)
	// renderer.gammaOutput = true
	renderer.gammaFactor = 2.2
	
	THREE.Cache.enabled = true
	
	const scene = new THREE.Scene()
	
	world.renderer = renderer
	world.scene = scene
}

export { init_canvas }