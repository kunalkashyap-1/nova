<script>
	import { goto } from '$app/navigation';
</script>

<div class="error-container">
	<div class="error-emoji">🛰️</div>
	<h1 class="error-title">404 - Page Not Found</h1>
	<p class="error-message">
		Sorry, the page you’re looking for doesn’t exist.<br />
		It may have been moved or deleted.
	</p>
	<button class="error-btn" on:click={() => goto('/chat', { replaceState: true })}>
		Go to Chat
	</button>
</div>

<style>
.error-container {
	display: flex;
	flex-direction: column;
	align-items: center;
	justify-content: center;
	height: 100vh;
	background: linear-gradient(135deg, #232526 0%, #414345 100%);
	color: #e5e7eb;
	text-align: center;
}
.error-emoji {
	font-size: 4rem;
	margin-bottom: 1.5rem;
}
.error-title {
	font-size: 2.5rem;
	font-weight: bold;
	margin-bottom: 1rem;
	color: #38bdf8;
	letter-spacing: 1px;
}
.error-message {
	font-size: 1.2rem;
	margin-bottom: 2rem;
	color: #cbd5e1;
}
.error-btn {
	background: linear-gradient(90deg, #22d3ee 0%, #38bdf8 100%);
	color: #232526;
	border: none;
	border-radius: 9999px;
	padding: 0.75rem 2.5rem;
	font-size: 1.1rem;
	font-weight: 600;
	cursor: pointer;
	transition: background 0.2s, color 0.2s, transform 0.2s;
	box-shadow: 0 2px 20px 0 #38bdf833;
}
.error-btn:hover {
	background: linear-gradient(90deg, #38bdf8 0%, #22d3ee 100%);
	color: #fff;
	transform: translateY(-2px) scale(1.03);
}
@media (max-width: 600px) {
	.error-title {
		font-size: 1.5rem;
	}
	.error-emoji {
		font-size: 2.5rem;
	}
	.error-btn {
		font-size: 1rem;
		padding: 0.6rem 1.5rem;
	}
}
</style>
